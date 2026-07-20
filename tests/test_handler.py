import base64
import json
from io import BytesIO
from unittest.mock import MagicMock

from PIL import Image

import handler


def encoded_image():
    b = BytesIO(); Image.new("RGB", (16, 16), "red").save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def job(**changes):
    data = {"person_image_base64": encoded_image(), "garment_image_base64": encoded_image(),
            "garment_category": "upper_body", "garment_description": "blue shirt", "seed": 42,
            "steps": 30, "guidance_scale": 2.0}
    data.update(changes); return {"input": data}


def test_invalid_before_model_import(monkeypatch):
    assert handler.handler({"input": {}})["error_type"] == "MISSING_FIELD"


def test_success(monkeypatch):
    bundle = MagicMock()
    monkeypatch.setattr("model_loader.get_model", lambda: (bundle, 1.25))
    monkeypatch.setattr("inference.run_inference", lambda bundle, request: Image.new("RGB", (768, 1024)))
    result = handler.handler(job())
    json.dumps(result)
    assert result["status"] == "completed"
    assert result["output"]["width"] == 768
    assert result["output"]["seed"] == 42


def test_inference_failure(monkeypatch):
    monkeypatch.setattr("model_loader.get_model", lambda: (MagicMock(), 0))
    monkeypatch.setattr("inference.run_inference", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    result = handler.handler(job())
    assert result == {"status": "error", "error_type": "RuntimeError", "message": "boom"}


def test_validation_cases():
    cases = [("person_image_base64", "%%%", "INVALID_BASE64"), ("garment_category", "hat", "INVALID_CATEGORY"),
             ("steps", 99, "INVALID_STEPS"), ("seed", 1.5, "INVALID_SEED"),
             ("guidance_scale", 99, "INVALID_GUIDANCE_SCALE")]
    for field, value, code in cases:
        assert handler.handler(job(**{field: value}))["error_type"] == code


def test_pipeline_and_list_outputs(monkeypatch):
    bundle = MagicMock()
    monkeypatch.setattr("model_loader.get_model", lambda: (bundle, 0))

    class PipelineOutput:
        images = [Image.new("RGB", (32, 24), "blue")]

    for output in ([Image.new("RGB", (20, 10))], PipelineOutput()):
        monkeypatch.setattr("inference.run_inference", lambda *_args, value=output: value)
        result = handler.handler(job())
        json.dumps(result)
        assert result["status"] == "completed"


def test_unsupported_result_is_safe_json_error(monkeypatch):
    monkeypatch.setattr("model_loader.get_model", lambda: (MagicMock(), 0))
    monkeypatch.setattr("inference.run_inference", lambda *_: object())
    result = handler.handler(job())
    json.dumps(result)
    assert result["status"] == "error"
    assert result["error_type"] == "TypeError"


def test_jpeg_quality_is_reduced_to_fit_response_limit(monkeypatch):
    qualities = []

    def encoded_at_quality(_image, quality):
        qualities.append(quality)
        encoded = "x" * (2000 if quality == 90 else 100)
        return encoded, 1500 if quality == 90 else 75

    monkeypatch.setattr(handler, "MAX_RUNPOD_RESPONSE_BYTES", 1000)
    monkeypatch.setattr(handler, "image_to_base64_with_size", encoded_at_quality)
    monkeypatch.setattr("model_loader.get_model", lambda: (MagicMock(), 0))
    monkeypatch.setattr("inference.run_inference", lambda *_: Image.new("RGB", (8, 8)))
    result = handler.handler(job())
    assert result["status"] == "completed"
    assert qualities == [90, 85]


def test_response_only_diagnostic_skips_input_and_model(monkeypatch):
    monkeypatch.setenv("RUNPOD_TEST_RESPONSE_ONLY", "true")
    result = handler.handler({})
    assert result == {
        "status": "completed",
        "diagnostic": True,
        "message": "RunPod response channel is working",
    }


def test_return_image_false_runs_inference_without_base64(monkeypatch):
    monkeypatch.setenv("RUNPOD_RETURN_IMAGE", "false")
    monkeypatch.setattr("model_loader.get_model", lambda: (MagicMock(), 0))
    monkeypatch.setattr(
        "inference.run_inference", lambda *_: Image.new("RGB", (32, 24), "blue")
    )
    result = handler.handler(job())
    assert result["diagnostic"] is True
    assert result["result_type"] == "Image"
    assert (result["width"], result["height"]) == (32, 24)
    assert "output" not in result


def test_image_is_resized_after_all_qualities_are_too_large(monkeypatch):
    attempts = []

    def encoded_for_size(image, quality):
        attempts.append((image.size, quality))
        encoded_size = 2000 if image.width > 1024 else 100
        return "x" * encoded_size, encoded_size * 3 // 4

    monkeypatch.setattr(handler, "MAX_RUNPOD_RESPONSE_BYTES", 1000)
    monkeypatch.setattr(handler, "image_to_base64_with_size", encoded_for_size)
    monkeypatch.setattr("model_loader.get_model", lambda: (MagicMock(), 0))
    monkeypatch.setattr(
        "inference.run_inference", lambda *_: Image.new("RGB", (2048, 2048))
    )
    result = handler.handler(job())
    assert attempts[: len(handler.JPEG_QUALITIES)] == [
        ((2048, 2048), quality) for quality in handler.JPEG_QUALITIES
    ]
    assert attempts[len(handler.JPEG_QUALITIES)] == ((1024, 1024), 90)
    assert (result["output"]["width"], result["output"]["height"]) == (1024, 1024)
