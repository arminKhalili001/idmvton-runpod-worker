import base64
from io import BytesIO
from PIL import Image, ImageOps


def fit_rgb(image: Image.Image, size=(768, 1024)) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def image_to_base64_with_size(image: Image.Image, quality: int = 90) -> tuple[str, int]:
    """Return a Base64 JPEG string and the original JPEG byte length."""
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
    jpeg_bytes = output.getvalue()
    return base64.b64encode(jpeg_bytes).decode("utf-8"), len(jpeg_bytes)


def image_to_base64(image: Image.Image, quality: int = 90) -> str:
    """Return a PIL image as a UTF-8-compatible Base64 JPEG string."""
    encoded, _ = image_to_base64_with_size(image, quality=quality)
    return encoded
