from __future__ import annotations

import threading
import time

from image_utils import image_to_base64
from schemas import InputError, parse_job

_inference_lock = threading.Lock()


def _failure(code, message):
    return {"status": "failed", "error": {"code": code, "message": message}}


def handler(job):
    total_started = time.perf_counter()
    try:
        request = parse_job(job)
    except InputError as exc:
        return _failure(exc.code, exc.message)
    try:
        with _inference_lock:
            from model_loader import get_model
            from inference import run_inference
            bundle, model_load_seconds = get_model()
            started = time.perf_counter()
            output = run_inference(bundle, request)
            inference_seconds = time.perf_counter() - started
            try:
                bundle.torch.cuda.empty_cache()
                bundle.torch.cuda.ipc_collect()
            except Exception:
                pass
        return {"status": "completed", "output": {"image_base64": image_to_base64(output),
            "mime_type": "image/jpeg", "width": output.width, "height": output.height, "seed": request.seed},
            "metrics": {"model_load_seconds": round(model_load_seconds, 3),
                        "inference_seconds": round(inference_seconds, 3),
                        "total_seconds": round(time.perf_counter() - total_started, 3)}}
    except Exception as exc:
        return _failure("INFERENCE_ERROR", str(exc))


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
