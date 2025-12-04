# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import numpy as np
from io import BytesIO
import base64

from image_store import ImageStore
import image_ops as ops

app = FastAPI(title="Whisper of Pixel API")

# CORS cho phép frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lúc deploy có thể thu hẹp domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ImageStore()

def get_image_or_404(image_id: str) -> np.ndarray:
    """
    Lấy ảnh từ store; nếu không có thì trả 404 thay vì văng KeyError.
    """
    try:
        return store.get(image_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Image not found or expired. Please upload again.",
        )
# ---------------------------
# Helper
# ---------------------------

def _validate_image_file(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")


def _uploadfile_to_numpy(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    try:
        pil_img = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image file")

    # Giữ ảnh màu (RGB); nếu gray thì convert thành 3 kênh
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return arr


def _response_from_images(labeled_imgs: List[tuple]):
    """
    labeled_imgs: list of (label, np.ndarray)
    Trả về JSON: { images: [ {label, data_url}, ...] }
    """
    items = []
    for label, img in labeled_imgs:
        data_url = store.to_data_url(img)
        items.append({"label": label, "data_url": data_url})
    return {"images": items}


# ---------------------------
# 0. Upload ảnh
# ---------------------------

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        pil_img = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    pil_rgb = pil_img.convert("RGB")
    img = np.array(pil_rgb)

    image_id = store.add(img)

    # dùng sẵn store.to_data_url để tạo preview PNG
    preview_url = store.to_data_url(img, fmt="PNG")

    return {
        "image_id": image_id,
        "preview_url": preview_url,
    }


# ---------------------------
# 1. COLOR REPRESENTATION
# ---------------------------

@app.get("/color/rgb")
async def color_rgb(image_id: str = Query(...)):
    img = store.get(image_id)
    r_img, g_img, b_img = ops.rgb_channels(img)
    return _response_from_images([
        ("R channel", r_img),
        ("G channel", g_img),
        ("B channel", b_img),
    ])


@app.get("/color/hsv")
async def color_hsv(image_id: str):
    img = store.get(image_id)
    h_img, s_img, v_img = ops.hsv_channels(img)
    return _response_from_images([
        ("Hue", h_img),
        ("Saturation", s_img),
        ("Value", v_img),
    ])


@app.get("/color/gray")
async def color_gray(image_id: str):
    img = store.get(image_id)
    gray = ops.to_gray(img)
    return _response_from_images([("Gray", gray)])


@app.get("/color/yuv")
async def color_yuv(image_id: str):
    img = store.get(image_id)
    y_img, u_img, v_img = ops.yuv_channels(img)
    return _response_from_images([
        ("Y channel", y_img),
        ("U channel", u_img),
        ("V channel", v_img),
    ])


# ---------------------------
# 2. POINT PROCESSING
# ---------------------------

@app.get("/point/negative")
async def point_negative(image_id: str):
    img = store.get(image_id)
    out = ops.negative(img)
    return _response_from_images([("Negative", out)])


@app.get("/point/threshold")
async def point_threshold(image_id: str, T: int = 128):
    img = store.get(image_id)
    out = ops.thresholding(img, T)
    return _response_from_images([("Threshold", out)])


@app.get("/point/log")
async def point_log(image_id: str, c: float = 1.0):
    img = store.get(image_id)
    out = ops.log_transform(img, c)
    return _response_from_images([("Log transform", out)])


@app.get("/point/gamma")
async def point_gamma(image_id: str, gamma: float = 1.0, c: float = 1.0):
    img = store.get(image_id)
    out = ops.gamma_transform(img, gamma, c)
    return _response_from_images([("Gamma transform", out)])


@app.get("/point/fuzzy-contrast")
async def point_fuzzy(image_id: str):
    img = store.get(image_id)
    out = ops.fuzzy_contrast_enhancement(img)
    return _response_from_images([("Fuzzy contrast", out)])


@app.get("/point/bit-plane")
async def point_bit_plane(image_id: str):
    img = store.get(image_id)
    planes = ops.bit_plane_slicing(img)
    labeled = [(f"Bit plane {k}", p) for k, p in enumerate(planes)]
    return _response_from_images(labeled)


@app.get("/point/bit-reconstruct")
async def point_bit_reconstruct(image_id: str, bits: str = "4,5,6,7"):
    img = store.get(image_id)
    bit_list = [int(b.strip()) for b in bits.split(",") if b.strip() != ""]
    out = ops.bit_plane_reconstruction(img, bit_list)
    return _response_from_images([("Reconstructed", out)])


# ---------------------------
# 3. HISTOGRAM
# ---------------------------

@app.get("/histogram/show")
async def hist_show(image_id: str):
    img = store.get(image_id)
    hist_img = ops.histogram_image(img)
    return _response_from_images([("Histogram", hist_img)])


@app.get("/histogram/equalization")
async def hist_equalization(image_id: str):
    img = store.get(image_id)
    eq = ops.histogram_equalization(img)
    return _response_from_images([("Equalized", eq)])


@app.get("/histogram/matching")
async def hist_matching(image_id: str, target_id: str):
    src = store.get(image_id)
    ref = store.get(target_id)
    matched = ops.histogram_matching(src, ref)
    return _response_from_images([("Histogram matched", matched)])


# ---------------------------
# 4. ADD NOISE
# ---------------------------

@app.get("/noise/gaussian")
async def noise_gaussian(image_id: str, mean: float = 0.0, var: float = 0.01):
    img = store.get(image_id)
    out = ops.add_gaussian_noise(img, mean, var)
    return _response_from_images([("Gaussian noise", out)])


@app.get("/noise/salt-pepper")
async def noise_salt_pepper(image_id: str, amount: float = 0.01, s_vs_p: float = 0.5):
    img = store.get(image_id)
    out = ops.add_salt_pepper_noise(img, amount, s_vs_p)
    return _response_from_images([("Salt & pepper noise", out)])


@app.get("/noise/uniform")
async def noise_uniform(image_id: str, low: float = -0.2, high: float = 0.2):
    img = store.get(image_id)
    out = ops.add_uniform_noise(img, low, high)
    return _response_from_images([("Uniform noise", out)])


@app.get("/noise/impulse")
async def noise_impulse(image_id: str, amount: float = 0.01):
    img = store.get(image_id)
    out = ops.add_impulse_noise(img, amount)
    return _response_from_images([("Impulse noise", out)])


# ---------------------------
# 5. SPATIAL FILTERS
# ---------------------------

@app.get("/spatial/correlation")
async def spatial_correlation(
    image_id: str = Query(...),
    target_id: str = Query(...),
):
    img = get_image_or_404(image_id)
    tpl = get_image_or_404(target_id)
    try:
        outs = ops.template_correlation(img, tpl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _response_from_images(outs)


@app.get("/spatial/convolution")
async def spatial_convolution(
    image_id: str = Query(...),
    target_id: str = Query(...),
):
    img = get_image_or_404(image_id)
    tpl = get_image_or_404(target_id)
    try:
        outs = ops.template_convolution(img, tpl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _response_from_images(outs)

@app.get("/spatial/mean")
async def spatial_mean(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.mean_filter(img, ksize)
    return _response_from_images([("Mean filter", out)])

@app.get("/spatial/gaussian")
async def spatial_gaussian(image_id: str, ksize: int = 3, sigma: float = 1.0):
    img = store.get(image_id)
    out = ops.gaussian_filter(img, ksize, sigma)
    return _response_from_images([("Gaussian filter", out)])

@app.get("/spatial/weighted-mean")
async def spatial_weighted(image_id: str):
    img = store.get(image_id)
    out = ops.weighted_mean_filter(img)
    return _response_from_images([("Weighted mean filter", out)])


@app.get("/spatial/median")
async def spatial_median(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.median_filter(img, ksize)
    return _response_from_images([("Median filter", out)])


@app.get("/spatial/max")
async def spatial_max(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.max_filter(img, ksize)
    return _response_from_images([("Max filter", out)])


@app.get("/spatial/min")
async def spatial_min(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.min_filter(img, ksize)
    return _response_from_images([("Min filter", out)])


@app.get("/spatial/sharpen")
async def spatial_sharpen(image_id: str):
    img = store.get(image_id)
    out = ops.laplacian_sharpen(img)
    return _response_from_images([("Laplacian sharpen", out)])


@app.get("/spatial/alpha-trimmed")
async def spatial_alpha_trimmed(image_id: str, ksize: int = 3, d: int = 2):
    img = store.get(image_id)
    out = ops.alpha_trimmed_mean_filter(img, ksize, d)
    return _response_from_images([("Alpha-trimmed mean", out)])


@app.get("/spatial/midpoint")
async def spatial_midpoint(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.midpoint_filter(img, ksize)
    return _response_from_images([("Midpoint filter", out)])


# ---------------------------
# 6. FREQUENCY FILTERS
# ---------------------------

@app.get("/frequency/fft2")
async def frequency_fft2(image_id: str):
    img = store.get(image_id)
    out = ops.fft2_magnitude_phase(img)
    return _response_from_images([("FFT magnitude", out)])


@app.get("/frequency/ilpf")
async def frequency_ilpf(image_id: str, D0: float = 30.0):
    img = store.get(image_id)
    out = ops.ilpf(img, D0)
    return _response_from_images([("ILPF", out)])


@app.get("/frequency/ihpf")
async def frequency_ihpf(image_id: str, D0: float = 30.0):
    img = store.get(image_id)
    out = ops.ihpf(img, D0)
    return _response_from_images([("IHPF", out)])


@app.get("/frequency/glpf")
async def frequency_glpf(image_id: str, D0: float = 30.0):
    img = store.get(image_id)
    out = ops.glpf(img, D0)
    return _response_from_images([("GLPF", out)])


@app.get("/frequency/ghpf")
async def frequency_ghpf(image_id: str, D0: float = 30.0):
    img = store.get(image_id)
    out = ops.ghpf(img, D0)
    return _response_from_images([("GHPF", out)])


@app.get("/frequency/blpf")
async def frequency_blpf(image_id: str, D0: float = 30.0, n: int = 2):
    img = store.get(image_id)
    out = ops.blpf(img, D0, n)
    return _response_from_images([("BLPF", out)])


@app.get("/frequency/bhpf")
async def frequency_bhpf(image_id: str, D0: float = 30.0, n: int = 2):
    img = store.get(image_id)
    out = ops.bhpf(img, D0, n)
    return _response_from_images([("BHPF", out)])


@app.get("/frequency/bandreject")
async def frequency_bandreject(image_id: str, D0: float = 50.0, W: float = 10.0):
    img = store.get(image_id)
    out = ops.bandreject_filter(img, D0, W)
    return _response_from_images([("Bandreject", out)])


@app.get("/frequency/bandpass")
async def frequency_bandpass(image_id: str, D0: float = 50.0, W: float = 10.0):
    img = store.get(image_id)
    out = ops.bandpass_filter(img, D0, W)
    return _response_from_images([("Bandpass", out)])


@app.get("/frequency/notch")
async def frequency_notch(image_id: str, u0: int = 30, v0: int = 30, D0: float = 10.0):
    img = store.get(image_id)
    out = ops.notch_filter(img, u0, v0, D0)
    return _response_from_images([("Notch reject", out)])


@app.get("/frequency/inverse")
async def frequency_inverse(image_id: str, D0: float = 30.0):
    img = store.get(image_id)
    out = ops.inverse_filter(img, D0)
    return _response_from_images([("Inverse filter", out)])


# ---------------------------
# 7. MORPHOLOGICAL
# ---------------------------

@app.get("/morph/erosion")
async def morph_erosion(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.morph_erosion(img, ksize)
    return _response_from_images([("Erosion", out)])


@app.get("/morph/dilation")
async def morph_dilation(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.morph_dilation(img, ksize)
    return _response_from_images([("Dilation", out)])


@app.get("/morph/opening")
async def morph_opening(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.morph_opening(img, ksize)
    return _response_from_images([("Opening", out)])


@app.get("/morph/closing")
async def morph_closing(image_id: str, ksize: int = 3):
    img = store.get(image_id)
    out = ops.morph_closing(img, ksize)
    return _response_from_images([("Closing", out)])


# ---------------------------
# 8. SEGMENTATION
# ---------------------------

@app.get("/seg/basic-global")
async def seg_basic_global(image_id: str, T: int = 128):
    img = store.get(image_id)
    out = ops.basic_global_threshold(img, T)
    return _response_from_images([("Basic global threshold", out)])


@app.get("/seg/otsu")
async def seg_otsu(image_id: str):
    img = store.get(image_id)
    out = ops.otsu_threshold(img)
    return _response_from_images([("Otsu threshold", out)])


@app.get("/seg/kmeans")
async def seg_kmeans(image_id: str, k: int = 3):
    img = store.get(image_id)
    out = ops.kmeans_segmentation(img, k)
    return _response_from_images([("K-means segmentation", out)])


# ---------------------------
# 9. PCA
# ---------------------------

@app.get("/pca/compress")
async def pca_compress(image_id: str, k: int = 50):
    img = store.get(image_id)
    out = ops.pca_compress_gray(img, k)
    return _response_from_images([("PCA compressed (gray)", out)])


# ---------------------------
# 10. YOLOv8 PERSON DETECTION
# ---------------------------

@app.get("/detect/yolov8")
async def detect_yolov8(image_id: str):
    img = store.get(image_id)
    out = ops.detect_person(img)
    return _response_from_images([("YOLOv8 person detection", out)])


# ---------------------------
# 11. COMPRESSION (HUFFMAN)
# ---------------------------

@app.get("/compression/huffman")
async def compression_huffman(image_id: str, ratio: int = 50):
    """
    ratio: % nén mong muốn (dùng để hiển thị, không đảm bảo chính xác).
    Trả về:
      - compressed (base64)
      - original_size
      - compressed_size
      - compression_ratio
      - rms (0 nếu lossless)
    """
    img = store.get(image_id)
    gray = ops.to_gray(img)
    original_bytes = gray.tobytes()
    compressed_bytes, codebook, padding = ops.huffman_encode(original_bytes)
    original_size = len(original_bytes)
    compressed_size = len(compressed_bytes)
    if original_size == 0:
        comp_ratio = 0.0
    else:
        comp_ratio = compressed_size / original_size

    rms = 0.0  # lossless

    compressed_b64 = base64.b64encode(compressed_bytes).decode("ascii")

    return {
        "rms": rms,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": comp_ratio,
        "target_ratio_percent": ratio,
        "compressed_data_base64": compressed_b64,
        "padding_bits": padding,
    }
