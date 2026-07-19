import base64
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
    assert handler.handler({"input": {}})["error"]["code"] == "MISSING_FIELD"


def test_success(monkeypatch):
    bundle = MagicMock()
    monkeypatch.setattr("model_loader.get_model", lambda: (bundle, 1.25))
    monkeypatch.setattr("inference.run_inference", lambda bundle, request: Image.new("RGB", (768, 1024)))
    result = handler.handler(job())
    assert result["status"] == "completed"
    assert result["output"]["width"] == 768
    assert result["output"]["seed"] == 42


def test_inference_failure(monkeypatch):
    monkeypatch.setattr("model_loader.get_model", lambda: (MagicMock(), 0))
    monkeypatch.setattr("inference.run_inference", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    result = handler.handler(job())
    assert result == {"status": "failed", "error": {"code": "INFERENCE_ERROR", "message": "boom"}}


def test_validation_cases():
    cases = [("person_image_base64", "%%%", "INVALID_BASE64"), ("garment_category", "hat", "INVALID_CATEGORY"),
             ("steps", 99, "INVALID_STEPS"), ("seed", 1.5, "INVALID_SEED"),
             ("guidance_scale", 99, "INVALID_GUIDANCE_SCALE")]
    for field, value, code in cases:
        assert handler.handler(job(**{field: value}))["error"]["code"] == code
