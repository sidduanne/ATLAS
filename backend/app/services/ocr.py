"""Local English OCR using Tesseract, with bounded execution and explicit outcomes."""
import io
import os
import subprocess
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass
class OCRResult:
    text: str
    status: str
    message: str | None = None


def extract_image_text(image: Image.Image) -> OCRResult:
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > 20_000_000:
        return OCRResult("", "failed", "OCR supports images up to 20 megapixels")
    # Transparent screenshots need a white background, not a black alpha discard.
    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    try:
        result = subprocess.run(
            [os.getenv("TESSERACT_CMD", "tesseract"), "stdin", "stdout", "-l", os.getenv("OCR_LANGUAGE", "eng"), "--psm", "3"],
            input=buffer.getvalue(), capture_output=True, timeout=60, check=False,
        )
    except FileNotFoundError:
        return OCRResult("", "unavailable", "Tesseract is not installed or TESSERACT_CMD is incorrect")
    except subprocess.TimeoutExpired:
        return OCRResult("", "timeout", "Text extraction exceeded 60 seconds; try a smaller or clearer image")
    except OSError:
        return OCRResult("", "failed", "Tesseract could not be started")
    if result.returncode:
        return OCRResult("", "failed", "Tesseract could not process the image; check OCR language data")
    text = result.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
    if not any(character.isalnum() for character in text):
        return OCRResult("", "no_text", "No readable text detected; the image is still available for visual search")
    return OCRResult(text, "extracted")
