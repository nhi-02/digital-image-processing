# image_store.py
import uuid
from typing import Dict
import numpy as np
from PIL import Image
from io import BytesIO
import base64


class ImageStore:
    """
    Lưu tất cả ảnh trong RAM dưới dạng numpy array (RGB hoặc Gray).
    image_id là UUID string.
    """

    def __init__(self):
        self._images: Dict[str, np.ndarray] = {}

    def add(self, img: np.ndarray) -> str:
        image_id = str(uuid.uuid4())
        self._images[image_id] = img
        return image_id

    def get(self, image_id: str) -> np.ndarray:
        if image_id not in self._images:
            raise KeyError(f"Image {image_id} not found")
        return self._images[image_id]

    def to_data_url(self, img: np.ndarray, fmt: str = "PNG") -> str:
        """
        Chuyển numpy array thành data URL "data:image/png;base64,..."
        """
        img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            mode = "L"
        else:
            mode = "RGB"

        pil_img = Image.fromarray(img, mode=mode)
        buf = BytesIO()
        pil_img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
