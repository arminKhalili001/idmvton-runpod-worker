from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 12 * 1024 * 1024
ALLOWED_CATEGORIES = {"upper_body", "lower_body", "dress"}
MIN_STEPS, MAX_STEPS = 10, 50
MIN_GUIDANCE, MAX_GUIDANCE = 1.0, 10.0


class InputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class TryOnInput:
    person_image: Image.Image
    garment_image: Image.Image
    garment_category: str
    garment_description: str
    seed: int
    steps: int
    guidance_scale: float


def _image(value: Any, field: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise InputError("MISSING_FIELD", f"{field} is required")
    value = value.strip()
    if value.startswith("data:"):
        parts = value.split(",", 1)
        if len(parts) != 2 or ";base64" not in parts[0]:
            raise InputError("INVALID_BASE64", f"{field} must be base64 encoded")
        value = parts[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise InputError("INVALID_BASE64", f"{field} is not valid base64") from None
    if not raw:
        raise InputError("INVALID_IMAGE", f"{field} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise InputError("IMAGE_TOO_LARGE", f"{field} exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.verify()
        with Image.open(BytesIO(raw)) as opened:
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise InputError("INVALID_IMAGE", f"{field} is not a readable image") from None


def parse_job(job: Any) -> TryOnInput:
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise InputError("INVALID_INPUT", "job.input must be an object")
    data = job["input"]
    person = _image(data.get("person_image_base64"), "person_image_base64")
    garment = _image(data.get("garment_image_base64"), "garment_image_base64")
    category = data.get("garment_category")
    if category not in ALLOWED_CATEGORIES:
        raise InputError("INVALID_CATEGORY", "garment_category must be upper_body, lower_body, or dress")
    description = data.get("garment_description")
    if not isinstance(description, str) or not description.strip() or len(description) > 500:
        raise InputError("INVALID_DESCRIPTION", "garment_description must be 1-500 characters")
    seed = data.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise InputError("INVALID_SEED", "seed must be an integer between 0 and 2147483647")
    steps = data.get("steps", 30)
    if isinstance(steps, bool) or not isinstance(steps, int) or not MIN_STEPS <= steps <= MAX_STEPS:
        raise InputError("INVALID_STEPS", f"steps must be an integer between {MIN_STEPS} and {MAX_STEPS}")
    guidance = data.get("guidance_scale", 2.0)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not MIN_GUIDANCE <= float(guidance) <= MAX_GUIDANCE:
        raise InputError("INVALID_GUIDANCE_SCALE", f"guidance_scale must be between {MIN_GUIDANCE} and {MAX_GUIDANCE}")
    return TryOnInput(person, garment, category, description.strip(), seed, steps, float(guidance))
