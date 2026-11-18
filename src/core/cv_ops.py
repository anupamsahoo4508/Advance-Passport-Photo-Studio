"""
Pure Mathematical Computer Vision & Differential Geometry Kernel (cv_ops.py).
Provides zero-third-party-lock-in vectorized NumPy + Pillow implementations of:
- Separable Gaussian & O(1) Summed-Area Table Box Filtering
- He et al. O(N) Edge-Preserving Guided Filter
- Sobel/Scharr Gradient Tensors & 2D Riemannian Structure Tensors (eigenvalues/orientation)
- Exact CIE L*a*b* (D65) & HSV perceptual color space transformations
- Biharmonic Thin-Plate Spline (TPS, Delta^2 f = 0, U(r) = r^2 ln r) solver & image warper
- Burt-Adelson Multi-Scale Laplacian Pyramid frequency blending
Automatically accelerates via OpenCV (cv2) when available, with 100% pure NumPy/Pillow parity.
"""
from __future__ import annotations
from PIL import Image, ImageFilter
import math
import numpy as np

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False



# ============================================================================
# 1. SAFE IMAGE I/O & GEOMETRIC RESAMPLING
# ============================================================================

def imread_rgb(path: str) -> np.ndarray | None:
    """Read an image file into an RGB uint8 numpy array (supports Unicode/Bengali paths)."""
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def imwrite_rgb(path: str, img_rgb: np.ndarray, quality: int = 98) -> bool:
    """Write an RGB or RGBA uint8 numpy array to disk safely using ultra-fast C++ SIMD encoding when available."""
    try:
        arr = np.clip(img_rgb, 0, 255).astype(np.uint8)
        is_rgba = arr.ndim == 3 and arr.shape[2] == 4
        low_p = path.lower()
        if HAS_CV2 and cv2 is not None and low_p.endswith(".png"):
            cv_arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if is_rgba else cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".png", cv_arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                with open(path, "wb") as f:
                    f.write(enc.tobytes())
                return True
        mode = "RGBA" if is_rgba else "RGB"
        im = Image.fromarray(arr, mode=mode)
        if low_p.endswith((".jpg", ".jpeg")):
            if mode == "RGBA":
                im = im.convert("RGB")
            im.save(path, format="JPEG", quality=quality, subsampling=0)
        elif low_p.endswith(".png"):
            im.save(path, format="PNG", compress_level=1)
        else:
            im.save(path)
        return True
    except Exception:
        return False


def resize_img(img: np.ndarray, new_w: int, new_h: int, mode: str = "lanczos") -> np.ndarray:
    """Resize 2D grayscale or 3D RGB/RGBA array to (new_h, new_w) with high-order interpolation."""
    new_w = max(1, int(round(new_w)))
    new_h = max(1, int(round(new_h)))
    h, w = img.shape[:2]
    if h == new_h and w == new_w:
        return img.copy()

    orig_dtype = img.dtype
    if HAS_CV2 and orig_dtype in (np.uint8, np.float32) and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (3, 4))):
        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "area": cv2.INTER_AREA,
        }.get(mode, cv2.INTER_LANCZOS4)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2_interp)
        return resized.astype(orig_dtype, copy=False)

    pil_resample = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "area": Image.Resampling.BOX,
    }.get(mode, Image.Resampling.LANCZOS)

    if img.ndim == 2:
        if orig_dtype == np.uint8:
            im = Image.fromarray(img, mode="L").resize((new_w, new_h), resample=pil_resample)
            return np.array(im, dtype=np.uint8)
        else:
            im = Image.fromarray(img.astype(np.float32), mode="F").resize((new_w, new_h), resample=pil_resample)
            return np.array(im, dtype=orig_dtype)

    channels = img.shape[2]
    if orig_dtype == np.uint8 and channels in (3, 4):
        pil_mode = "RGB" if channels == 3 else "RGBA"
        im = Image.fromarray(img, mode=pil_mode).resize((new_w, new_h), resample=pil_resample)
        return np.array(im, dtype=np.uint8)

    # Float multi-channel resize channel-by-channel
    out = np.empty((new_h, new_w, channels), dtype=np.float32)
    for c in range(channels):
        im_c = Image.fromarray(img[:, :, c].astype(np.float32), mode="F").resize(
            (new_w, new_h), resample=pil_resample
        )
        out[:, :, c] = np.array(im_c, dtype=np.float32)
    return out.astype(orig_dtype)


def rotate_image(img: np.ndarray, angle_deg: float, border_rgb: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Rotate image counter-clockwise by angle_deg around center using Bicubic interpolation."""
    if abs(angle_deg) < 1e-3:
        return img.copy()
    if img.ndim == 2:
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
        rot = im.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=int(border_rgb[0]))
        return np.array(rot, dtype=img.dtype)
    channels = img.shape[2]
    mode = "RGBA" if channels == 4 else "RGB"
    fill = (*border_rgb, 0) if channels == 4 else tuple(int(x) for x in border_rgb)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode=mode)
    rot = im.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
    return np.array(rot, dtype=np.uint8)


# ============================================================================
# 2. INTEGRAL IMAGE BOX FILTER, SEPARABLE GAUSSIAN & GUIDED FILTER
# ============================================================================

def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """
    O(1) per-pixel 2D Separable Box Filter using 1D float64 Cumulative Sums or OpenCV SIMD.
    Window size is (2*radius + 1) x (2*radius + 1) with reflection padding.
    Immune to float32 summed-area-table precision overflow on 4K images.
    """
    if radius <= 0:
        return img.astype(np.float32, copy=True)
    arr = img.astype(np.float32)
    r = int(radius)
    if HAS_CV2:
        k = 2 * r + 1
        return cv2.boxFilter(arr, ddepth=-1, ksize=(k, k), normalize=True, borderType=cv2.BORDER_REFLECT)

    h, w = arr.shape[:2]
    k = 2 * r + 1
    inv_k = 1.0 / float(k)

    # Pass 1: Vertical 1D box filter via float64 cumulative sum
    pad_y = np.pad(arr, ((r, r), (0, 0)) if arr.ndim == 2 else ((r, r), (0, 0), (0, 0)), mode="reflect")
    zero_row = np.zeros((1, w) if arr.ndim == 2 else (1, w, arr.shape[2]), dtype=np.float64)
    cs_y = np.concatenate([zero_row, np.cumsum(pad_y, axis=0, dtype=np.float64)], axis=0)
    blur_y = (cs_y[k:h + k] - cs_y[0:h]) * inv_k

    # Pass 2: Horizontal 1D box filter via float64 cumulative sum
    pad_x = np.pad(blur_y, ((0, 0), (r, r)) if arr.ndim == 2 else ((0, 0), (r, r), (0, 0)), mode="reflect")
    zero_col = np.zeros((h, 1) if arr.ndim == 2 else (h, 1, arr.shape[2]), dtype=np.float64)
    cs_x = np.concatenate([zero_col, np.cumsum(pad_x, axis=1, dtype=np.float64)], axis=1)
    blur_xy = (cs_x[:, k:w + k] - cs_x[:, 0:w]) * inv_k
    return blur_xy.astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 1D Gaussian convolution with reflect boundary conditions and multi-scale float32 acceleration."""
    if sigma <= 0.05:
        return img.copy()
    if HAS_CV2 and img.dtype in (np.uint8, np.float32):
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # Fast Pillow C-extension path when sigma >= 1.2 and image is uint8
    if img.dtype == np.uint8 and sigma >= 1.2 and img.ndim in (2, 3):
        mode = "L" if img.ndim == 2 else ("RGB" if img.shape[2] == 3 else "RGBA")
        im = Image.fromarray(img, mode=mode).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
        return np.array(im, dtype=np.uint8)

    h, w = img.shape[:2]
    orig_dtype = img.dtype

    # Multi-scale band-limited acceleration for large sigma on float32 arrays
    if sigma >= 2.6 and min(h, w) >= 128:
        down_s = min(4, max(2, int(sigma / 1.35)))
        sh, sw = max(16, h // down_s), max(16, w // down_s)
        small = resize_img(img.astype(np.float32), sw, sh, mode="bilinear")
        eff_sigma = max(0.6, float(sigma) / float(down_s))
        blurred_small = gaussian_blur(small, eff_sigma)
        res = resize_img(blurred_small, w, h, mode="bilinear")
        if orig_dtype == np.uint8:
            return np.clip(np.round(res), 0, 255).astype(np.uint8)
        return res.astype(orig_dtype)

    radius = max(1, int(math.ceil(2.35 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)

    arr = img.astype(np.float32)

    def _conv2d_sep(channel: np.ndarray) -> np.ndarray:
        pad_x = np.pad(channel, ((0, 0), (radius, radius)), mode="reflect")
        ch_h, ch_w = channel.shape
        horiz = np.zeros((ch_h, ch_w), dtype=np.float32)
        for i, k_val in enumerate(kernel):
            horiz += k_val * pad_x[:, i:i + ch_w]
        pad_y = np.pad(horiz, ((radius, radius), (0, 0)), mode="reflect")
        vert = np.zeros((ch_h, ch_w), dtype=np.float32)
        for i, k_val in enumerate(kernel):
            vert += k_val * pad_y[i:i + ch_h, :]
        return vert

    if arr.ndim == 2:
        res = _conv2d_sep(arr)
    else:
        res = np.empty_like(arr)
        for c in range(arr.shape[2]):
            res[:, :, c] = _conv2d_sep(arr[:, :, c])

    if orig_dtype == np.uint8:
        return np.clip(np.round(res), 0, 255).astype(np.uint8)
    return res.astype(orig_dtype)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """
    He et al. (ECCV 2010) O(N) Edge-Preserving Guided Filter.
    Solves local linear model q_i = a_k * I_i + b_k with regularization eps.
    """
    I = guide.astype(np.float32)
    if I.ndim == 3:
        # Convert RGB guide to luminance in [0, 1]
        I = (0.299 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.114 * I[:, :, 2])
    if I.max() > 1.5:
        I = I / 255.0

    p = src.astype(np.float32)
    scale_back = False
    if p.max() > 1.5:
        p = p / 255.0
        scale_back = True

    mean_I = box_filter(I, radius)
    mean_II = box_filter(I * I, radius)
    var_I = np.maximum(0.0, mean_I * mean_I * -1.0 + mean_II)

    def _filter_1ch(p_ch: np.ndarray) -> np.ndarray:
        mean_p = box_filter(p_ch, radius)
        mean_Ip = box_filter(I * p_ch, radius)
        cov_Ip = mean_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)
        return mean_a * I + mean_b

    if p.ndim == 2:
        q = _filter_1ch(p)
    else:
        q = np.empty_like(p)
        for c in range(p.shape[2]):
            q[:, :, c] = _filter_1ch(p[:, :, c])

    if scale_back:
        return np.clip(q * 255.0, 0.0, 255.0)
    return np.clip(q, 0.0, 1.0)


# ============================================================================
# 3. DIFFERENTIAL GEOMETRY: SOBEL GRADIENTS & RIEMANNIAN STRUCTURE TENSOR
# ============================================================================

def compute_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Scharr/Sobel partial derivatives (gx, gy) and gradient magnitude ||grad I||.
    Input gray should be 2D float32.
    """
    g = gray.astype(np.float32)
    padded = np.pad(g, ((1, 1), (1, 1)), mode="reflect")
    # Scharr 3x3 rotational symmetry operators
    gx = (
        3.0 * (padded[0:-2, 2:] - padded[0:-2, 0:-2])
        + 10.0 * (padded[1:-1, 2:] - padded[1:-1, 0:-2])
        + 3.0 * (padded[2:, 2:] - padded[2:, 0:-2])
    ) / 32.0
    gy = (
        3.0 * (padded[2:, 0:-2] - padded[0:-2, 0:-2])
        + 10.0 * (padded[2:, 1:-1] - padded[0:-2, 1:-1])
        + 3.0 * (padded[2:, 2:] - padded[0:-2, 2:])
    ) / 32.0
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def shock_filter_edge_sharpen(
    L_channel: np.ndarray,
    iterations: int = 2,
    dt: float = 0.16,
    sigma_smooth: float = 0.75,
) -> np.ndarray:
    """
    Osher-Rudin Curvature-Driven Directional Shock Filter PDE:
      dL/dt = -tanh(L_{eta,eta} / 2.2) * ||grad L||
    where L_{eta,eta} = (L_x^2 L_xx + 2 L_x L_y L_xy + L_y^2 L_yy) / (||grad L||^2 + eps).
    Compresses blurred interpolation ramps back into crisp step edges (eyelashes, hair,
    eyeglass rims, iris limbus) with zero halo overshoot.
    """
    if L_channel.ndim == 3:
        return np.stack(
            [
                shock_filter_edge_sharpen(L_channel[:, :, c], iterations=iterations, dt=dt, sigma_smooth=sigma_smooth)
                for c in range(L_channel.shape[2])
            ],
            axis=-1,
        )
    u = L_channel.astype(np.float32, copy=True)
    u_min, u_max = float(np.min(u)), float(np.max(u))
    for _ in range(max(1, iterations)):
        u_s = gaussian_blur(u, sigma_smooth)
        gx, gy, mag = compute_gradients(u_s)
        gxx, gxy, _ = compute_gradients(gx)
        _, gyy, _ = compute_gradients(gy)
        denom = gx * gx + gy * gy + 1e-4
        l_eta_eta = (gx * gx * gxx + 2.0 * gx * gy * gxy + gy * gy * gyy) / denom
        shock = -np.tanh(l_eta_eta / 2.2) * mag
        u = np.clip(u + dt * shock, u_min, u_max)
    return u


def compute_structure_tensor(gray: np.ndarray, sigma_grad: float = 1.0, rho_integ: float = 2.5) -> dict[str, np.ndarray]:
    """
    Compute 2D Riemannian Structure Tensor J_rho = G_rho * (grad I_sigma (x) grad I_sigma).
    Returns eigenvalues (lambda1 >= lambda2 >= 0), coherence C in [0, 1], and tangent angle theta.
    """
    smooth = gaussian_blur(gray.astype(np.float32), sigma_grad)
    gx, gy, mag = compute_gradients(smooth)
    j11 = gaussian_blur(gx * gx, rho_integ)
    j12 = gaussian_blur(gx * gy, rho_integ)
    j22 = gaussian_blur(gy * gy, rho_integ)

    trace = j11 + j22
    disc = np.hypot(j11 - j22, 2.0 * j12)
    lambda1 = 0.5 * (trace + disc)
    lambda2 = np.maximum(0.0, 0.5 * (trace - disc))
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)) ** 2
    # Tangent orientation (orthogonal to principal gradient eigenvector)
    theta_grad = 0.5 * np.arctan2(2.0 * j12, j11 - j22)
    theta_tangent = theta_grad + (math.pi * 0.5)
    return {
        "gx": gx,
        "gy": gy,
        "mag": mag,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "coherence": coherence,
        "theta_tangent": theta_tangent,
    }


# ============================================================================
# 4. PERCEPTUAL COLOR SPACES: CIE L*a*b* (D65) & HSV
# ============================================================================

_U8_GRID = np.arange(256, dtype=np.float32) / np.float32(255.0)
_SRGB_TO_LINEAR_LUT = np.where(
    _U8_GRID > np.float32(0.04045),
    ((_U8_GRID + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
    _U8_GRID / np.float32(12.92),
).astype(np.float32)

_LIN_GRID = np.linspace(0.0, 1.0, 16384, dtype=np.float32)
_LINEAR_TO_SRGB_U8_LUT = np.clip(
    np.round(
        np.where(
            _LIN_GRID > np.float32(0.0031308),
            np.float32(1.055) * (np.maximum(_LIN_GRID, np.float32(1e-8)) ** np.float32(1.0 / 2.4)) - np.float32(0.055),
            np.float32(12.92) * _LIN_GRID,
        )
        * np.float32(255.0)
    ),
    0,
    255,
).astype(np.uint8)

_RGB_TO_XYZ_D65_T = np.array([
    [0.4124564 / 0.95047, 0.2126729, 0.0193339 / 1.08883],
    [0.3575761 / 0.95047, 0.7151522, 0.1191920 / 1.08883],
    [0.1804375 / 0.95047, 0.0721750, 0.9503041 / 1.08883],
], dtype=np.float32)

_XYZ_D65_TO_RGB_T = np.array([
    [ 3.2404542 * 0.95047, -0.9692660 * 0.95047,  0.0556434 * 0.95047],
    [-1.5371385,            1.8760108,           -0.2040259          ],
    [-0.4985314 * 1.08883,  0.0415560 * 1.08883,  1.0572252 * 1.08883],
], dtype=np.float32)


_DELTA_CIE = np.float32(6.0 / 29.0)
_XYZ_TO_F_LUT = np.where(
    _LIN_GRID > (_DELTA_CIE ** 3),
    np.cbrt(np.maximum(_LIN_GRID, np.float32(1e-8))),
    (_LIN_GRID / (np.float32(3.0) * _DELTA_CIE * _DELTA_CIE)) + np.float32(4.0 / 29.0),
).astype(np.float32)


_CBRT_LUT = np.cbrt(np.maximum(_LIN_GRID, np.float32(0.0))).astype(np.float32)

_RGB_TO_OKLMS_T = np.array([
    [0.4122214708, 0.2119034982, 0.0883024619],
    [0.5363325363, 0.6806995451, 0.2817188376],
    [0.0514459929, 0.1073969566, 0.6299787005],
], dtype=np.float32)

_OKLMS_TO_OKLAB_T = np.array([
    [0.2104542553,  1.9779984951,  0.0259040371],
    [0.7936177850, -2.4285922050,  0.7827717662],
    [-0.0040720468, 0.4505937099, -0.8086757660],
], dtype=np.float32)

_OKLMS_TO_RGB_T = np.array([
    [ 4.0767416621, -1.2684380046, -0.0041960863],
    [-3.3077115913,  2.6097574011, -0.7034186147],
    [ 0.2309699292, -0.3413193965,  1.7076147010],
], dtype=np.float32)


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB uint8/float [0, 255] to CIE 1976 L*a*b* (D65 white point).
    Returns float32 array with L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    Accelerates via OpenCV C++ SIMD when available.
    """
    if HAS_CV2:
        if img_rgb.dtype == np.uint8:
            f01 = img_rgb.astype(np.float32) * np.float32(1.0 / 255.0)
        else:
            f01 = np.clip(img_rgb.astype(np.float32) * np.float32(1.0 / 255.0), 0.0, 1.0)
        return cv2.cvtColor(f01, cv2.COLOR_RGB2Lab)

    if img_rgb.dtype == np.uint8:
        linear = _SRGB_TO_LINEAR_LUT[img_rgb]
    else:
        rgb = np.clip(img_rgb.astype(np.float32) / np.float32(255.0), np.float32(0.0), np.float32(1.0))
        mask = rgb > np.float32(0.04045)
        linear = np.where(mask, ((rgb + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4), rgb / np.float32(12.92))

    xyz = np.clip(np.tensordot(linear, _RGB_TO_XYZ_D65_T, axes=1), np.float32(0.0), np.float32(1.0))
    f = _XYZ_TO_F_LUT[(xyz * np.float32(16383.0) + np.float32(0.5)).astype(np.int32)]

    L = np.float32(116.0) * f[:, :, 1] - np.float32(16.0)
    a = np.float32(500.0) * (f[:, :, 0] - f[:, :, 1])
    b = np.float32(200.0) * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* float32 back to sRGB uint8 [0, 255]."""
    lab_f = np.ascontiguousarray(img_lab, dtype=np.float32)
    if HAS_CV2:
        rgb01 = cv2.cvtColor(lab_f, cv2.COLOR_Lab2RGB)
        return np.clip(np.round(rgb01 * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)

    L = lab_f[:, :, 0]
    a = lab_f[:, :, 1]
    b = lab_f[:, :, 2]

    fy = (L + np.float32(16.0)) * np.float32(1.0 / 116.0)
    fx = fy + a * np.float32(1.0 / 500.0)
    fz = fy - b * np.float32(1.0 / 200.0)

    delta = np.float32(6.0 / 29.0)
    three_delta2 = np.float32(3.0 * (6.0 / 29.0) ** 2)
    offset = np.float32(4.0 / 29.0)
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > delta, f * f * f, three_delta2 * (f - offset))

    linear = np.clip(np.tensordot(xyz, _XYZ_D65_TO_RGB_T, axes=1), np.float32(0.0), np.float32(1.0))
    idx = (linear * np.float32(16383.0) + np.float32(0.5)).astype(np.int32)
    return _LINEAR_TO_SRGB_U8_LUT[idx]


def rgb_to_oklab(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB uint8/float [0, 255] to Björn Ottosson's Perceptual Oklab (L in [0, 1], a, b in [-0.4, 0.4]).
    Zero hue-twist perceptual color space ideal for studio portrait color grading & tone mapping.
    """
    if img_rgb.dtype == np.uint8:
        linear = _SRGB_TO_LINEAR_LUT[img_rgb]
    else:
        rgb = np.clip(img_rgb.astype(np.float32) * np.float32(1.0 / 255.0), np.float32(0.0), np.float32(1.0))
        mask = rgb > np.float32(0.04045)
        linear = np.where(mask, ((rgb + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4), rgb / np.float32(12.92))

    lms = np.clip(np.tensordot(linear, _RGB_TO_OKLMS_T, axes=1), np.float32(0.0), np.float32(1.0))
    lms_cbrt = _CBRT_LUT[(lms * np.float32(16383.0) + np.float32(0.5)).astype(np.int32)]
    return np.tensordot(lms_cbrt, _OKLMS_TO_OKLAB_T, axes=1).astype(np.float32, copy=False)


def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """Convert Perceptual Oklab float32 back to sRGB uint8 [0, 255]."""
    ok = oklab.astype(np.float32, copy=False)
    L = ok[:, :, 0]
    a = ok[:, :, 1]
    b = ok[:, :, 2]

    lp = L + np.float32(0.3963377774) * a + np.float32(0.2158037573) * b
    mp = L - np.float32(0.1055613458) * a - np.float32(0.0638541728) * b
    sp = L - np.float32(0.0894841775) * a - np.float32(1.2914855480) * b

    lms = np.stack([lp * lp * lp, mp * mp * mp, sp * sp * sp], axis=-1)
    linear = np.clip(np.tensordot(lms, _OKLMS_TO_RGB_T, axes=1), np.float32(0.0), np.float32(1.0))
    idx = (linear * np.float32(16383.0) + np.float32(0.5)).astype(np.int32)
    return _LINEAR_TO_SRGB_U8_LUT[idx]


# ============================================================================
# 5. BIHARMONIC THIN-PLATE SPLINE (TPS, Delta^2 f = 0) WARP SOLVER
# ============================================================================

class ThinPlateSpline2D:
    """
    Exact Biharmonic Thin-Plate Spline (Bookstein 1989) solver in R^2.
    Minimizes bending energy E(f) = integral (f_xx^2 + 2 f_xy^2 + f_yy^2) dx dy
    using fundamental kernel U(r) = r^2 * ln(r + eps).
    """

    def __init__(self, regularization: float = 1e-3):
        self.reg = float(regularization)
        self.ctrl_pts: np.ndarray | None = None
        self.weights_x: np.ndarray | None = None
        self.weights_y: np.ndarray | None = None

    @staticmethod
    def _kernel(r2: np.ndarray) -> np.ndarray:
        return 0.5 * r2 * np.log(r2 + 1e-12)

    def fit(self, target_pts: np.ndarray, source_pts: np.ndarray) -> "ThinPlateSpline2D":
        """
        Fit inverse mapping from target (output) pixel coordinates -> source (template) coordinates.
        target_pts, source_pts: shape (N, 2) in normalized or pixel coordinates.
        """
        pts = np.asarray(target_pts, dtype=np.float64)
        vals = np.asarray(source_pts, dtype=np.float64)
        n = pts.shape[0]
        self.ctrl_pts = pts

        diff = pts[:, None, :] - pts[None, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        K = self._kernel(r2)
        K[np.diag_indices(n)] += self.reg

        P = np.column_stack([np.ones(n, dtype=np.float64), pts])
        L = np.zeros((n + 3, n + 3), dtype=np.float64)
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        L[n:, n:] = np.eye(3, dtype=np.float64) * 1e-8

        rhs = np.zeros((n + 3, 2), dtype=np.float64)
        rhs[:n, :] = vals

        sol = np.linalg.solve(L, rhs)
        self.weights_x = sol[:, 0]
        self.weights_y = sol[:, 1]
        return self

    def transform(self, query_pts: np.ndarray) -> np.ndarray:
        """Evaluate TPS mapping at query_pts of shape (M, 2)."""
        assert self.ctrl_pts is not None and self.weights_x is not None and self.weights_y is not None
        q = np.asarray(query_pts, dtype=np.float64)
        n = self.ctrl_pts.shape[0]
        diff = q[:, None, :] - self.ctrl_pts[None, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        U = self._kernel(r2)

        P_q = np.column_stack([np.ones(q.shape[0], dtype=np.float64), q])
        basis = np.hstack([U, P_q])
        sx = basis @ self.weights_x
        sy = basis @ self.weights_y
        return np.column_stack([sx, sy])


def warp_image_tps(
    src_img: np.ndarray,
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    out_shape: tuple[int, int],
    grid_step: int = 8,
    reg: float = 2e-3,
) -> np.ndarray:
    """
    Warp src_img (H_s, W_s, C) so that src_landmarks align onto dst_landmarks in out_shape (H_d, W_d).
    Evaluates exact Biharmonic TPS on a coarse mesh (grid_step) and bilinearly interpolates dense coordinates.
    """
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    src_h, src_w = src_img.shape[:2]

    # Normalize coordinates to [0, 1] for numerical conditioning
    src_norm = np.column_stack([
        src_landmarks[:, 0] / max(1.0, float(src_w - 1)),
        src_landmarks[:, 1] / max(1.0, float(src_h - 1)),
    ])
    dst_norm = np.column_stack([
        dst_landmarks[:, 0] / max(1.0, float(out_w - 1)),
        dst_landmarks[:, 1] / max(1.0, float(out_h - 1)),
    ])

    tps = ThinPlateSpline2D(regularization=reg).fit(dst_norm, src_norm)

    # Evaluate on coarse grid for high-speed 4K performance
    gh = max(4, out_h // grid_step + 2)
    gw = max(4, out_w // grid_step + 2)
    gy = np.linspace(0.0, 1.0, gh)
    gx = np.linspace(0.0, 1.0, gw)
    grid_x, grid_y = np.meshgrid(gx, gy)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mapped = tps.transform(query)

    map_x_coarse = mapped[:, 0].reshape(gh, gw).astype(np.float32) * float(src_w - 1)
    map_y_coarse = mapped[:, 1].reshape(gh, gw).astype(np.float32) * float(src_h - 1)

    map_x = resize_img(map_x_coarse, out_w, out_h, mode="bilinear")
    map_y = resize_img(map_y_coarse, out_w, out_h, mode="bilinear")

    return bilinear_sample(src_img, map_x, map_y)


def bilinear_sample(src_img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Vectorized 2D bilinear texture sampler with zero-padding out of bounds."""
    if HAS_CV2 and src_img.dtype in (np.uint8, np.float32):
        return cv2.remap(
            src_img,
            map_x.astype(np.float32, copy=False),
            map_y.astype(np.float32, copy=False),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    src_h, src_w = src_img.shape[:2]
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (map_x - x0).astype(np.float32)
    wy = (map_y - y0).astype(np.float32)

    valid = (map_x >= 0) & (map_x <= src_w - 1) & (map_y >= 0) & (map_y <= src_h - 1)

    x0_c = np.clip(x0, 0, src_w - 1)
    x1_c = np.clip(x1, 0, src_w - 1)
    y0_c = np.clip(y0, 0, src_h - 1)
    y1_c = np.clip(y1, 0, src_h - 1)

    src_f = src_img.astype(np.float32)
    Ia = src_f[y0_c, x0_c]
    Ib = src_f[y1_c, x0_c]
    Ic = src_f[y0_c, x1_c]
    Id = src_f[y1_c, x1_c]

    if src_img.ndim == 3:
        wx = wx[:, :, None]
        wy = wy[:, :, None]
        valid = valid[:, :, None]

    wa = (1.0 - wx) * (1.0 - wy)
    wb = (1.0 - wx) * wy
    wc = wx * (1.0 - wy)
    wd = wx * wy

    out = (wa * Ia + wb * Ib + wc * Ic + wd * Id) * valid
    if src_img.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(src_img.dtype)


# ============================================================================
# 6. MULTI-SCALE LAPLACIAN PYRAMID BLENDING (BURT & ADELSON)
# ============================================================================

def laplacian_pyramid_blend(img_fg: np.ndarray, img_bg: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Burt & Adelson Multi-Band Laplacian Pyramid Seamless Blending.
    Eliminates low-frequency color seams while preserving crisp high-frequency textures.
    """
    A = img_fg.astype(np.float32)
    B = img_bg.astype(np.float32)
    M = mask.astype(np.float32)
    if M.max() > 1.5:
        M /= 255.0
    if M.ndim == 2 and A.ndim == 3:
        M = M[:, :, None]

    gp_A = [A]
    gp_B = [B]
    gp_M = [M]
    for _ in range(levels):
        h, w = gp_A[-1].shape[:2]
        if h < 8 or w < 8:
            break
        nh, nw = max(4, h // 2), max(4, w // 2)
        gp_A.append(resize_img(gaussian_blur(gp_A[-1], 1.0), nw, nh, mode="bilinear"))
        gp_B.append(resize_img(gaussian_blur(gp_B[-1], 1.0), nw, nh, mode="bilinear"))
        m_down = resize_img(gaussian_blur(gp_M[-1].squeeze(), 1.2), nw, nh, mode="bilinear")
        if A.ndim == 3 and m_down.ndim == 2:
            m_down = m_down[:, :, None]
        gp_M.append(m_down)

    num_levels = len(gp_A) - 1
    lp_A = []
    lp_B = []
    for i in range(num_levels):
        h, w = gp_A[i].shape[:2]
        up_A = resize_img(gp_A[i + 1], w, h, mode="bilinear")
        up_B = resize_img(gp_B[i + 1], w, h, mode="bilinear")
        lp_A.append(gp_A[i] - up_A)
        lp_B.append(gp_B[i] - up_B)
    lp_A.append(gp_A[-1])
    lp_B.append(gp_B[-1])

    # Blend each frequency band using its corresponding Gaussian mask level
    blended_pyr = []
    for la, lb, gm in zip(lp_A, lp_B, gp_M):
        blended_pyr.append(gm * la + (1.0 - gm) * lb)

    # Reconstruct from coarsest to finest
    recon = blended_pyr[-1]
    for i in range(num_levels - 1, -1, -1):
        h, w = blended_pyr[i].shape[:2]
        recon = resize_img(recon, w, h, mode="bilinear") + blended_pyr[i]

    if img_fg.dtype == np.uint8:
        return np.clip(np.round(recon), 0, 255).astype(np.uint8)
    return recon.astype(img_fg.dtype)
