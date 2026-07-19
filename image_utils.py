import base64
from io import BytesIO
from PIL import Image, ImageOps


def fit_rgb(image: Image.Image, size=(768, 1024)) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def image_to_base64(image: Image.Image, quality: int = 95) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")
