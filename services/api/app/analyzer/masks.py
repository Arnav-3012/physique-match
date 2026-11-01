"""Strict mask decoding: the server-side half of ADR-003. Anything that is not a binary mask is
rejected with ValueError (the route turns it into a 422)."""

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from ml.src.preprocess import H, W, from_browser


def decode_mask(b64: str) -> np.ndarray:
    """Base64 PNG -> clean bool (H, W) mask. Size is checked from the header, before any pixels."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
    except (binascii.Error, UnidentifiedImageError, ValueError) as e:
        raise ValueError("mask is not a valid base64 PNG") from e
    if img.format != "PNG":
        raise ValueError("mask must be a PNG")
    if img.size != (W, H):
        raise ValueError(f"mask must be {W}x{H}, got {img.size[0]}x{img.size[1]}")
    if img.mode != "L":
        raise ValueError(f"mask must be 1-channel 8-bit, got mode {img.mode}")
    pixels = np.asarray(img)
    if not np.isin(pixels, (0, 255)).all():
        raise ValueError("mask must contain only the values 0 and 255")
    if pixels.all():
        raise ValueError("mask is entirely white")
    return from_browser(pixels == 255)
