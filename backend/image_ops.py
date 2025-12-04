# image_ops.py
import numpy as np
import cv2
from skimage import exposure, morphology
from sklearn.cluster import KMeans

# ---------------------------
# Helper chung
# ---------------------------

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    img = ensure_rgb(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def _apply_per_channel(img: np.ndarray, func):
    """
    Helper: áp dụng func cho từng kênh nếu ảnh màu, hoặc 1 lần nếu ảnh xám.
    func: nhận 2D array (H,W), trả về 2D array.
    """
    if img.ndim == 2:
        return func(img)

    b, g, r = cv2.split(img)
    b2 = func(b)
    g2 = func(g)
    r2 = func(r)
    return cv2.merge([b2, g2, r2])


# ---------------------------
# 1. COLOR REPRESENTATION
# ---------------------------

def rgb_channels(img: np.ndarray):
    """
    Phân tích cấu tạo RGB:
    - Input phải là ảnh màu 3 kênh (RGB).
    - Trả về 3 ảnh màu:
        + R channel: vùng sáng thể hiện giá trị R (G=B=0)
        + G channel: vùng sáng thể hiện giá trị G (R=B=0)
        + B channel: vùng sáng thể hiện giá trị B (R=G=0)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("RGB channels representation is only available for color images.")

    # giả sử img hiện đang ở dạng RGB (vì mình convert từ PIL -> RGB khi upload)
    r, g, b = cv2.split(img)
    zeros = np.zeros_like(r)

    # tạo ảnh màu cho từng kênh
    red_vis   = cv2.merge([r, zeros, zeros])   # chỉ kênh R
    green_vis = cv2.merge([zeros, g, zeros])   # chỉ kênh G
    blue_vis  = cv2.merge([zeros, zeros, b])   # chỉ kênh B

    return [red_vis, green_vis, blue_vis]


def hsv_channels(img: np.ndarray):
    """
    Phân tích cấu tạo HSV:
    - Input phải là ảnh màu 3 kênh (RGB).
    - Trả về 3 ảnh màu pseudo-color dùng colormap HSV:
        + Hue: dùng giá trị H, tô bằng COLORMAP_HSV
        + Saturation: dùng giá trị S, tô bằng COLORMAP_HSV
        + Value: dùng giá trị V, tô bằng COLORMAP_HSV
    Giống style: imshow(channel, cmap='hsv') trong Matplotlib.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("HSV representation is only available for color images.")

    # RGB -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Chuẩn hóa 3 kênh về [0,255] cho rõ tương phản
    h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ánh xạ colormap HSV (pseudo-color)
    h_color_bgr = cv2.applyColorMap(h_norm, cv2.COLORMAP_HSV)
    s_color_bgr = cv2.applyColorMap(s_norm, cv2.COLORMAP_HSV)
    v_color_bgr = cv2.applyColorMap(v_norm, cv2.COLORMAP_HSV)

    # OpenCV dùng BGR -> đổi về RGB để trả cho frontend
    h_color_rgb = cv2.cvtColor(h_color_bgr, cv2.COLOR_BGR2RGB)
    s_color_rgb = cv2.cvtColor(s_color_bgr, cv2.COLOR_BGR2RGB)
    v_color_rgb = cv2.cvtColor(v_color_bgr, cv2.COLOR_BGR2RGB)

    return [h_color_rgb, s_color_rgb, v_color_rgb]



def yuv_channels(img: np.ndarray):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("YUV representation is only available for color images (RGB input).")

    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)

    y_vis = cv2.merge([y, y, y])
    u_vis = cv2.merge([u, u, u])
    v_vis = cv2.merge([v, v, v])

    return [y_vis, u_vis, v_vis]


# ---------------------------
# 2. POINT PROCESSING
# ---------------------------

def negative(img: np.ndarray) -> np.ndarray:
    """
    Negative:
    - Xám: 255 - g
    - Màu: 255 - R,G,B
    """
    inv = 255 - img.astype(np.uint8)
    return inv


def thresholding(img: np.ndarray, T: int = 128) -> np.ndarray:
    """
    Threshold trên kênh sáng (gray) để ra binary.
    """
    gray = to_gray(img)
    _, dst = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    return dst


def log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Log transform trên từng kênh (nếu ảnh màu).
    """
    img_f = img.astype(np.float32)
    norm = img_f / 255.0
    log_img = c * np.log1p(norm)
    max_val = log_img.max() if log_img.max() > 0 else 1.0
    log_img = log_img / max_val * 255.0
    return np.clip(log_img, 0, 255).astype(np.uint8)


def gamma_transform(img: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Gamma (power-law) trên từng kênh.
    """
    img_f = img.astype(np.float32)
    norm = img_f / 255.0
    gamma_img = c * np.power(norm, gamma)
    max_val = gamma_img.max() if gamma_img.max() > 0 else 1.0
    gamma_img = gamma_img / max_val * 255.0
    return np.clip(gamma_img, 0, 255).astype(np.uint8)


def fuzzy_contrast_enhancement(img: np.ndarray) -> np.ndarray:
    """
    Fuzzy rule-based contrast enhancement:
    - Tăng cường contrast trên kênh độ sáng, sau đó ghép lại thành ảnh màu.
    """
    rgb = ensure_rgb(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    norm = gray / 255.0

    # membership functions
    dark = np.clip((0.5 - norm) / 0.5, 0, 1)
    bright = np.clip((norm - 0.5) / 0.5, 0, 1)
    medium = 1.0 - np.maximum(dark, bright)

    out = norm.copy()
    out = out - 0.3 * dark * out              # darker
    out = out + 0.3 * bright * (1.0 - out)    # brighter
    out = out + 0.15 * medium * (out - 0.5)   # contrast around mid

    out = np.clip(out, 0, 1)
    enhanced_gray = (out * 255.0).astype(np.uint8)

    # Dùng enhanced_gray như kênh V của HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = enhanced_gray
    hsv_enh = cv2.merge([h, s, v])
    rgb_enh = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2RGB)
    return rgb_enh


def bit_plane_slicing(img: np.ndarray):
    """
    Trả về list 8 ảnh bit-plane của ảnh gray.
    (bit-plane làm trên kênh sáng)
    """
    gray = to_gray(img)
    planes = []
    for k in range(8):
        plane = ((gray >> k) & 1) * 255
        planes.append(plane.astype(np.uint8))
    return planes


def bit_plane_reconstruction(img: np.ndarray, bits):
    """
    bits: list các bit index cần dùng để reconstruct (0..7).
    """
    gray = to_gray(img)
    recon = np.zeros_like(gray, dtype=np.uint16)
    for k in bits:
        plane = (gray >> k) & 1
        recon += (plane * (2 ** k)).astype(np.uint16)
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    return recon


# ---------------------------
# 3. HISTOGRAM & EQUALIZATION / MATCHING
# ---------------------------

def histogram_image(img: np.ndarray, width: int = 256, height: int = 256) -> np.ndarray:
    """
    Vẽ histogram (trên gray) thành ảnh 256x256 đen trắng.
    """
    gray = to_gray(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.max() + 1e-6)
    h_img = np.full((height, width), 255, dtype=np.uint8)
    for x in range(256):
        h = int(hist[x] * (height - 1))
        cv2.line(h_img, (x, height - 1), (x, height - 1 - h), 0, 1)
    return h_img


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Equalization trên kênh Y (luminance), giữ màu.
    """
    rgb = ensure_rgb(img)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    rgb_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    return rgb_eq


def histogram_matching(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Histogram matching trên kênh Y (luminance), giữ màu.
    """
    src_rgb = ensure_rgb(src)
    ref_rgb = ensure_rgb(ref)

    src_ycc = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2YCrCb)
    ref_ycc = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2YCrCb)
    y_src, cr_src, cb_src = cv2.split(src_ycc)
    y_ref, _, _ = cv2.split(ref_ycc)

    # skimage match_histograms cho 2D
    y_matched = exposure.match_histograms(y_src, y_ref)
    y_matched = np.clip(y_matched, 0, 255).astype(np.uint8)

    ycc_out = cv2.merge([y_matched, cr_src, cb_src])
    rgb_out = cv2.cvtColor(ycc_out, cv2.COLOR_YCrCb2RGB)
    return rgb_out


# ---------------------------
# 4. NOISE ADDING (per-channel)
# ---------------------------

def add_gaussian_noise(img: np.ndarray, mean=0.0, var=0.01) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.normal(mean, np.sqrt(var), img_f.shape)
    noisy = img_f + noise
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, amount=0.01, s_vs_p=0.5) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    num_pixels = h * w
    num_salt = int(np.ceil(amount * num_pixels * s_vs_p))
    num_pepper = int(np.ceil(amount * num_pixels * (1.0 - s_vs_p)))

    # Salt
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    if img.ndim == 2:
        out[coords] = 255
    else:
        out[coords[0], coords[1], :] = 255

    # Pepper
    coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
    if img.ndim == 2:
        out[coords] = 0
    else:
        out[coords[0], coords[1], :] = 0

    return out


def add_uniform_noise(img: np.ndarray, low=-0.2, high=0.2) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.uniform(low, high, img_f.shape)
    noisy = img_f + noise
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)


def add_impulse_noise(img: np.ndarray, amount=0.01) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    num_imp = int(amount * h * w)

    coords = (np.random.randint(0, h, num_imp), np.random.randint(0, w, num_imp))
    values = np.random.choice([0, 255], num_imp)

    if img.ndim == 2:
        out[coords] = values
    else:
        out[coords[0], coords[1], :] = values[:, None]

    return out


# ---------------------------
# 5. SPATIAL FILTERS (per-channel)
# ---------------------------

def correlation(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Correlation trên kênh sáng (theo đúng lý thuyết).
    """
    gray = to_gray(img)
    k = kernel.astype(np.float32)
    dst = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
    return dst


def convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution = correlation với kernel lật.
    """
    k = np.flipud(np.fliplr(kernel))
    return correlation(img, k)


def mean_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    def _f(ch):
        return cv2.blur(ch, (ksize, ksize))
    return _apply_per_channel(img, _f)

def gaussian_filter(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing (lọc Gauss) per-channel:
    - Nếu ảnh xám: GaussianBlur 1 kênh
    - Nếu ảnh màu: áp dụng cho từng kênh R,G,B
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1  # kernel size phải lẻ

    def _f(ch):
        return cv2.GaussianBlur(ch, (ksize, ksize), sigmaX=float(sigma))

    return _apply_per_channel(img, _f)

def weighted_mean_filter(img: np.ndarray) -> np.ndarray:
    """
    Weighted 3x3:
    [1 2 1
     2 4 2
     1 2 1] / 16
    """
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0

    def _f(ch):
        return cv2.filter2D(ch, -1, kernel)
    return _apply_per_channel(img, _f)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    def _f(ch):
        return cv2.medianBlur(ch, ksize)
    return _apply_per_channel(img, _f)


def max_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)

    def _f(ch):
        return cv2.dilate(ch, kernel)
    return _apply_per_channel(img, _f)


def min_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)

    def _f(ch):
        return cv2.erode(ch, kernel)
    return _apply_per_channel(img, _f)


def laplacian_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Sharpen từng kênh: ch_out = ch - Laplacian(ch).
    """
    def _f(ch):
        lap = cv2.Laplacian(ch, cv2.CV_32F, ksize=3)
        sharp = ch.astype(np.float32) - lap
        sharp = np.clip(sharp, 0, 255)
        return sharp.astype(np.uint8)

    return _apply_per_channel(img, _f)


def midpoint_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    pad = ksize // 2

    def _f(ch):
        padded = cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        h, w = ch.shape
        out = np.zeros_like(ch, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+ksize, j:j+ksize]
                out[i, j] = 0.5 * (np.min(region) + np.max(region))
        return out.astype(np.uint8)

    return _apply_per_channel(img, _f)


def alpha_trimmed_mean_filter(img: np.ndarray, ksize: int = 3, d: int = 2) -> np.ndarray:
    """
    Alpha-trimmed mean filter trên từng kênh.
    """
    pad = ksize // 2
    n = ksize * ksize
    trim = d // 2

    def _f(ch):
        padded = cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        h, w = ch.shape
        out = np.zeros_like(ch, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+ksize, j:j+ksize].flatten()
                region = np.sort(region)
                trimmed = region[trim:n-trim]
                out[i, j] = np.mean(trimmed)
        return out.astype(np.uint8)

    return _apply_per_channel(img, _f)


# ---------------------------
# 6. FREQUENCY DOMAIN FILTERS (trên gray)
# ---------------------------

def _center_transform(f):
    return np.fft.fftshift(f)


def _inverse_center_transform(f):
    return np.fft.ifftshift(f)


def _fft2_gray(img: np.ndarray):
    gray = to_gray(img).astype(np.float32)
    F = np.fft.fft2(gray)
    return gray, F


def fft2_magnitude_phase(img: np.ndarray):
    gray, F = _fft2_gray(img)
    F_shift = _center_transform(F)
    magnitude = np.log1p(np.abs(F_shift))
    magnitude = magnitude / (magnitude.max() + 1e-6) * 255.0
    return magnitude.astype(np.uint8)


def _create_distance_matrix(shape):
    M, N = shape
    u = np.arange(M) - M / 2
    v = np.arange(N) - N / 2
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt(U**2 + V**2)
    return D


def ilpf(img: np.ndarray, D0: float = 30.0) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = (D <= D0).astype(np.float32)
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def ihpf(img: np.ndarray, D0: float = 30.0) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = (D > D0).astype(np.float32)
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def glpf(img: np.ndarray, D0: float = 30.0) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = np.exp(-(D**2) / (2 * (D0**2)))
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def ghpf(img: np.ndarray, D0: float = 30.0) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def blpf(img: np.ndarray, D0: float = 30.0, n: int = 2) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = 1 / (1 + (D / D0) ** (2 * n))
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def bhpf(img: np.ndarray, D0: float = 30.0, n: int = 2) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = 1 - 1 / (1 + (D / D0) ** (2 * n))
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def bandreject_filter(img: np.ndarray, D0: float = 50.0, W: float = 10.0) -> np.ndarray:
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = 1 / (1 + ((D * W) / (D**2 - D0**2 + 1e-5)) ** 2)
    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def bandpass_filter(img: np.ndarray, D0: float = 50.0, W: float = 10.0) -> np.ndarray:
    br = bandreject_filter(img, D0, W)
    gray = to_gray(img).astype(np.float32)
    bp = gray.max() - br.astype(np.float32)
    bp = np.clip(bp, 0, 255)
    return bp.astype(np.uint8)


def notch_filter(img: np.ndarray, u0: int = 30, v0: int = 30, D0: float = 10.0) -> np.ndarray:
    """
    Notch reject filter đối xứng tại (+-u0, +-v0) quanh tâm.
    """
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    U, V = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    u_c, v_c = M / 2, N / 2

    H = np.ones((M, N), dtype=np.float32)

    def notch(u_k, v_k):
        Dk = np.sqrt((U - (u_c + u_k)) ** 2 + (V - (v_c + v_k)) ** 2)
        Dk_sym = np.sqrt((U - (u_c - u_k)) ** 2 + (V - (v_c - v_k)) ** 2)
        return (1 / (1 + (D0 / (Dk + 1e-5)) ** 2)) * (1 / (1 + (D0 / (Dk_sym + 1e-5)) ** 2))

    H *= notch(u0, v0)

    G = _inverse_center_transform(_center_transform(F) * H)
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


def inverse_filter(img: np.ndarray, D0: float = 30.0) -> np.ndarray:
    """
    Lọc nghịch đảo đơn giản với giả định H(u,v) là Gaussian LPF.
    """
    gray, F = _fft2_gray(img)
    M, N = gray.shape
    D = _create_distance_matrix((M, N))
    H = np.exp(-(D**2) / (2 * (D0**2)))
    eps = 1e-3
    G = _inverse_center_transform(_center_transform(F) / (H + eps))
    g = np.abs(np.fft.ifft2(G))
    g = g / (g.max() + 1e-6) * 255.0
    return g.astype(np.uint8)


# ---------------------------
# 7. MORPHOLOGICAL (gray)
# ---------------------------

def morph_erosion(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = to_gray(img)
    kernel = morphology.square(ksize).astype(np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)
    return eroded


def morph_dilation(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = to_gray(img)
    kernel = morphology.square(ksize).astype(np.uint8)
    dil = cv2.dilate(gray, kernel, iterations=1)
    return dil


def morph_opening(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = to_gray(img)
    kernel = morphology.square(ksize).astype(np.uint8)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return opened


def morph_closing(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = to_gray(img)
    kernel = morphology.square(ksize).astype(np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return closed


# ---------------------------
# 8. SEGMENTATION (gray)
# ---------------------------

def basic_global_threshold(img: np.ndarray, T: int = 128) -> np.ndarray:
    return thresholding(img, T)


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    _, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dst


def kmeans_segmentation(img: np.ndarray, k: int = 3) -> np.ndarray:
    gray = to_gray(img)
    Z = gray.reshape((-1, 1)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=0)
    labels = kmeans.fit_predict(Z)
    centers = np.uint8(kmeans.cluster_centers_.flatten())
    segmented = centers[labels].reshape(gray.shape)
    return segmented


# ---------------------------
# 9. PCA (simple compression trên gray)
# ---------------------------

def pca_compress_gray(img: np.ndarray, num_components: int = 50) -> np.ndarray:
    gray = to_gray(img).astype(np.float32)
    mean = np.mean(gray, axis=0, keepdims=True)
    X = gray - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(num_components, Vt.shape[0])
    Xk = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    recon = Xk + mean
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    return recon


# ---------------------------
# 10. YOLOv8 PERSON DETECTION
# ---------------------------

from ultralytics import YOLO

_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_person(img: np.ndarray) -> np.ndarray:
    """
    Nhận diện person và vẽ box lên ảnh RGB.
    """
    model = get_yolo_model()
    rgb = ensure_rgb(img)
    results = model.predict(source=rgb, conf=0.25, verbose=False)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb


# ---------------------------
# 11. COMPRESSION (HUFFMAN + RMS)
# ---------------------------

import heapq
from collections import Counter, namedtuple
from typing import Dict, Tuple

HuffmanNode = namedtuple("HuffmanNode", ["freq", "symbol", "left", "right"])

def build_huffman_tree(data: bytes) -> HuffmanNode:
    heap = []
    counter = Counter(data)
    for sym, freq in counter.items():
        heapq.heappush(heap, (freq, id(sym), HuffmanNode(freq, sym, None, None)))

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(f1+f2, None, n1, n2)
        heapq.heappush(heap, (merged.freq, id(merged), merged))

    return heap[0][2]


def build_codes(node: HuffmanNode, prefix: str = "", codebook: Dict[int, str] = None) -> Dict[int, str]:
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix or "0"
    else:
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(data: bytes) -> Tuple[bytes, Dict[int, str], int]:
    """
    Trả về: compressed_bytes, codebook, padding_bits
    """
    if not data:
        return b"", {}, 0
    root = build_huffman_tree(data)
    codebook = build_codes(root)
    bits = "".join(codebook[b] for b in data)
    padding = (8 - len(bits) % 8) % 8
    bits += "0" * padding
    out_bytes = int(bits, 2).to_bytes(len(bits) // 8, byteorder="big")
    return out_bytes, codebook, padding


def rms_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    mse = np.mean(diff ** 2)
    return float(np.sqrt(mse))
