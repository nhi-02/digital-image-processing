# image_store.py
import uuid
from typing import Dict
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from collections import OrderedDict


class ImageStore:
    """
    Lưu ảnh trong RAM với giới hạn:
    - Tối đa 10 ảnh.
    - Khi thêm ảnh thứ 11 -> xóa ảnh cũ nhất (FIFO: first-in, first-out).

    Ảnh được lưu dạng numpy array.
    """

    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self._images: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def add(self, img: np.ndarray) -> str:
        """
        Thêm một ảnh mới vào store.
        Nếu quá max_items -> xóa ảnh cũ nhất.
        """
        image_id = str(uuid.uuid4())

        # Nếu đã đầy, pop ảnh đầu tiên (ảnh cũ nhất)
        if len(self._images) >= self.max_items:
            self._images.popitem(last=False)   # last=False => pop ảnh cũ nhất

        # Thêm ảnh mới vào cuối (mới nhất)
        self._images[image_id] = img
        return image_id

    def get(self, image_id: str) -> np.ndarray:
        """
        Lấy ảnh theo ID.
        """
        if image_id not in self._images:
            raise KeyError(f"Image {image_id} not found")

        # Không đổi thứ tự FIFO khi get
        return self._images[image_id]

    def to_data_url(self, img: np.ndarray, fmt: str = "PNG") -> str:
        """
        Chuyển numpy array thành data URL dạng:
        "data:image/png;base64,..."
        """
        img = np.clip(img, 0, 255).astype(np.uint8)

        mode = "L" if img.ndim == 2 else "RGB"

        pil_img = Image.fromarray(img, mode=mode)
        buf = BytesIO()
        pil_img.save(buf, format=fmt)

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"

        return f"data:{mime};base64,{b64}"
