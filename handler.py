from __future__ import annotations

import json
import os
import threading
import time
import traceback

from PIL import Image

from image_utils import image_to_base64_with_size
from schemas import InputError, parse_job

_inference_lock = threading.Lock()
RUNPOD_API_LIMIT_BYTES = 10_000_000
DEFAULT_SAFE_RESPONSE_BYTES = 7_000_000
MAX_RUNPOD_RESPONSE_BYTES = int(
    os.getenv("MAX_RUNPOD_RESPONSE_BYTES", "7000000")
)
if MAX_RUNPOD_RESPONSE_BYTES > 8_000_000:
    print(
        "warning: MAX_RUNPOD_RESPONSE_BYTES exceeds 8000000; clamping to 8000000",
        flush=True,
    )
    MAX_RUNPOD_RESPONSE_BYTES = 8_000_000
JPEG_QUALITIES = (90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10)
RESIZE_MAX_DIMENSIONS = ((1024, 1536), (896, 1344), (768, 1152), (640, 960))


def _env_flag(name, default):
    return os.getenv(name, "true" if default else "false").strip().lower() == "true"


def _error_response(error_type, message):
    response = {"status": "error", "error_type": str(error_type), "message": str(message)}
    json.dumps(response, ensure_ascii=False)
    print(f"returning error response: error_type={response['error_type']}", flush=True)
    return response


def _extract_image(result):
    if isinstance(result, Image.Image):
        return result
    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("Inference returned an empty image list")
        return _extract_image(result[0])
    images = getattr(result, "images", None)
    if images is not None:
        return _extract_image(images)
    raise TypeError(f"Unsupported inference result type: {type(result).__name__}")


def _completed_response(result, request, model_load_seconds, inference_seconds, total_started):
    image = _extract_image(result).convert("RGB")
    print(f"extracted image dimensions: width={image.width}, height={image.height}", flush=True)
    metrics = {
        "model_load_seconds": round(float(model_load_seconds), 3),
        "inference_seconds": round(float(inference_seconds), 3),
        "total_seconds": round(float(time.perf_counter() - total_started), 3),
    }
    if not _env_flag("RUNPOD_RETURN_IMAGE", True):
        response = {
            "status": "completed",
            "diagnostic": True,
            "result_type": type(result).__name__,
            "width": int(image.width),
            "height": int(image.height),
            "metrics": metrics,
        }
        serialized = json.dumps(response, ensure_ascii=False)
        response_size = len(serialized.encode("utf-8"))
        print(f"serialized response byte size: {response_size}", flush=True)
        print("immediately before returning response", flush=True)
        return response

    last_size = 0
    images_to_try = [image]
    for max_size in RESIZE_MAX_DIMENSIONS:
        resized = image.copy()
        resized.thumbnail(max_size, Image.Resampling.LANCZOS)
        if resized.size != images_to_try[-1].size:
            images_to_try.append(resized)

    for candidate in images_to_try:
        for quality in JPEG_QUALITIES:
            print(f"JPEG quality attempted: {quality}", flush=True)
            encoded, jpeg_size = image_to_base64_with_size(candidate, quality=quality)
            base64_size = len(encoded.encode("utf-8"))
            response = {
                "status": "completed",
                "output": {
                    "image_base64": encoded,
                    "mime_type": "image/jpeg",
                    "width": int(candidate.width),
                    "height": int(candidate.height),
                    "seed": int(request.seed),
                },
                "metrics": metrics,
            }
            serialized = json.dumps(response, ensure_ascii=False)
            last_size = len(serialized.encode("utf-8"))
            print(f"raw JPEG byte size: {jpeg_size}", flush=True)
            print(f"Base64 byte size: {base64_size}", flush=True)
            print(f"serialized response byte size: {last_size}", flush=True)
            print(
                f"quality={quality}, jpeg_bytes={jpeg_size}, "
                f"base64_bytes={len(encoded.encode('utf-8'))}, "
                f"response_bytes={last_size}",
                flush=True,
            )
            if last_size <= MAX_RUNPOD_RESPONSE_BYTES:
                print(f"selected JPEG quality: {quality}", flush=True)
                print(f"final response size: {last_size}", flush=True)
                print("immediately before returning response", flush=True)
                return response
    raise ValueError(
        f"Encoded response is {last_size / 1_000_000:.3f} MB and exceeds the "
        f"configured {MAX_RUNPOD_RESPONSE_BYTES / 1_000_000:.3f} MB limit"
    )


def handler(job):
    total_started = time.perf_counter()
    print("handler received job", flush=True)
    try:
        if _env_flag("RUNPOD_TEST_RESPONSE_ONLY", False):
            response = {
                "status": "completed",
                "diagnostic": True,
                "message": "RunPod response channel is working",
            }
            serialized = json.dumps(response, ensure_ascii=False)
            response_size = len(serialized.encode("utf-8"))
            print(f"serialized response byte size: {response_size}", flush=True)
            print("immediately before returning response", flush=True)
            return response

        request = parse_job(job)
        print("input parsed", flush=True)
        with _inference_lock:
            from model_loader import get_model
            from inference import run_inference
            print("loading model", flush=True)
            bundle, model_load_seconds = get_model()
            print("model loaded", flush=True)
            started = time.perf_counter()
            print("starting inference", flush=True)
            output = run_inference(bundle, request)
            inference_seconds = time.perf_counter() - started
            print("inference finished", flush=True)
            print(f"raw result Python type: {type(output).__name__}", flush=True)
            try:
                bundle.torch.cuda.empty_cache()
                bundle.torch.cuda.ipc_collect()
            except Exception:
                pass
        return _completed_response(
            output, request, model_load_seconds, inference_seconds, total_started
        )
    except Exception as exc:
        traceback.print_exc()
        error_type = exc.code if isinstance(exc, InputError) else type(exc).__name__
        return _error_response(error_type, str(exc))


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
