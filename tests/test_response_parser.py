import base64
import json
from io import BytesIO
from PIL import Image
from image_utils import image_to_base64, image_to_base64_with_size


def test_encoded_response_is_json_serializable_and_decodable():
    encoded = image_to_base64(Image.new("RGB", (768, 1024), "white"))
    response = {"status": "completed", "output": {"image_base64": encoded, "mime_type": "image/jpeg",
                "width": 768, "height": 1024, "seed": 42}}
    json.dumps(response)
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (768, 1024)


def test_image_to_base64_converts_rgba_to_jpeg():
    encoded = image_to_base64(Image.new("RGBA", (8, 6), (255, 0, 0, 128)))
    assert isinstance(encoded, str)
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGB"
        assert image.format == "JPEG"


def test_image_to_base64_with_size_reports_jpeg_buffer_length():
    encoded, jpeg_size = image_to_base64_with_size(Image.new("RGB", (8, 6), "red"))
    assert jpeg_size == len(base64.b64decode(encoded))
