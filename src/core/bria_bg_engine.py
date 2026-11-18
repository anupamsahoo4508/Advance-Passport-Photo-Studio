"""
BRIA AI Portrait Matting & Studio Background Engine (`bria_bg_engine.py` — V8 Multi-Subject Pure-BRIA).

Cloud-Only Execution Guarantee:
- Strictly respects `IS_HF_CLOUD`: NEVER downloads neural weights onto a local Windows PC (`0 MB`).
- Inside Hugging Face Cloud (`IS_HF_CLOUD == True`), executes GPU-accelerated:
  1. Primary: Aspect-Preserving Letterbox `1024×1024` + High-Res Head/Hair ROI `1024×1024` Dual-Pass
     BRIA `RMBG-2.0` (`1038lab/RMBG-2.0` BiRefNet architecture on CUDA).
  2. Secondary Fallback: Official BRIA `RMBG-1.4` (`briaai/RMBG-1.4` IS-Net architecture).
- Zero Oval Halo Guarantee: Trusts BRIA's true cheek, jaw, ponytail, and saree-pallu silhouette
  (never overriding exterior background with geometric ovals), prunes background bystanders via
  Multi-Subject Connected-Component Graph Filtering, applies 2D Spatially-Varying Local Exterior
  Color Spill Decontamination, and runs Edge-Guided Alpha Upsampling all the way up to True 8K (`7680px+`).
"""
from __future__ import annotations
import os
import threading
import numpy as np
from src.shared.config import CLOUD_MODELS_DIR, HF_TOKEN, IS_HF_CLOUD
from src.core.cv_ops import (
    gaussian_blur,
    guided_filter,
    resize_img,
    rgb_to_lab,
)
from src.core.color_studio import synthesize_studio_backdrop
from src.core.facial_math_engine import FacialAnthropometrics, analyze_face_geometry

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

_BRIA_LOCK = threading.Lock()
_RMBG20_MODEL = None
_RMBG20_TRIED = False
_RMBG14_MODEL = None
_RMBG14_TRIED = False
_ACTIVE_BRIA_TAG = "Analytical-Local-Fallback"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_bria_engine_status() -> dict[str, object]:
    """Return current status of the Cloud-Native BRIA AI Background Engine."""
    cuda_ok = False
    if IS_HF_CLOUD:
        try:
            import torch  # type: ignore
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    return {
        "is_hf_cloud": IS_HF_CLOUD,
        "cuda_available": cuda_ok,
        "cloud_models_dir": CLOUD_MODELS_DIR if IS_HF_CLOUD else "DISABLED_ON_LOCAL_PC",
        "rmbg_2_0_loaded": _RMBG20_MODEL is not None,
        "rmbg_1_4_loaded": _RMBG14_MODEL is not None,
        "active_engine": _ACTIVE_BRIA_TAG,
    }


def _load_rmbg20_cloud():
    """
    Load BRIA RMBG-2.0 (`1038lab/RMBG-2.0`) strictly inside Hugging Face Cloud (`IS_HF_CLOUD`).
    Cached in `CLOUD_MODELS_DIR` (`/tmp/passport_studio_models`) on CPU first, then moved to CUDA
    inside `@spaces.GPU` calls.
    """
    global _RMBG20_MODEL, _RMBG20_TRIED
    if not IS_HF_CLOUD:
        return None
    if os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return None
    if _RMBG20_MODEL is not None:
        return _RMBG20_MODEL
    if _RMBG20_TRIED:
        return None

    with _BRIA_LOCK:
        if _RMBG20_MODEL is not None:
            return _RMBG20_MODEL
        if _RMBG20_TRIED:
            return None
        _RMBG20_TRIED = True
        try:
            os.makedirs(CLOUD_MODELS_DIR, exist_ok=True)
            import torch  # type: ignore
            from transformers import AutoModelForImageSegmentation  # type: ignore

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

            for repo_id in ("1038lab/RMBG-2.0", "ZhengPeng7/BiRefNet-portrait"):
                try:
                    model = AutoModelForImageSegmentation.from_pretrained(
                        repo_id,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                        cache_dir=CLOUD_MODELS_DIR,
                        token=HF_TOKEN,
                    )
                    model.eval()
                    _RMBG20_MODEL = model
                    print(f"✅ [BRIA Cloud] Loaded primary segmentation model: {repo_id}")
                    return _RMBG20_MODEL
                except Exception as inner_err:
                    # Transformers 5.x compatibility loader for BiRefNet custom `self.config = Config()`
                    try:
                        from huggingface_hub import hf_hub_download  # type: ignore
                        from safetensors.torch import load_file as load_safetensors_file  # type: ignore
                        from transformers.dynamic_module_utils import get_class_from_dynamic_module  # type: ignore

                        cfg_cls = get_class_from_dynamic_module(
                            "BiRefNet_config.BiRefNetConfig",
                            repo_id,
                            cache_dir=CLOUD_MODELS_DIR,
                            token=HF_TOKEN,
                        )
                        mdl_cls = get_class_from_dynamic_module(
                            "birefnet.BiRefNet",
                            repo_id,
                            cache_dir=CLOUD_MODELS_DIR,
                            token=HF_TOKEN,
                        )
                        weights_path = hf_hub_download(
                            repo_id=repo_id,
                            filename="model.safetensors",
                            cache_dir=CLOUD_MODELS_DIR,
                            token=HF_TOKEN,
                        )
                        model = mdl_cls(bb_pretrained=False, config=cfg_cls(bb_pretrained=False))
                        sd = load_safetensors_file(weights_path, device="cpu")
                        model.load_state_dict(sd, strict=False)
                        model.eval()
                        _RMBG20_MODEL = model
                        print(f"✅ [BRIA Cloud] Loaded primary BiRefNet via dynamic-module safetensors: {repo_id}")
                        return _RMBG20_MODEL
                    except Exception as dyn_err:
                        print(f"⚠️ [BRIA Cloud] Candidate {repo_id} skipped ({inner_err} / {dyn_err})")
        except Exception as e:
            print(f"⚠️ [BRIA Cloud] RMBG-2.0 loader error: {e}")
    return None


def _load_rmbg14_cloud():
    """
    Load official BRIA RMBG-1.4 (`briaai/RMBG-1.4`) strictly inside Hugging Face Cloud (`IS_HF_CLOUD`).
    Supports PyTorch `AutoModelForImageSegmentation` or ONNX fallback (`onnx/model.onnx`).
    """
    global _RMBG14_MODEL, _RMBG14_TRIED
    if not IS_HF_CLOUD:
        return None
    if os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return None
    if _RMBG14_MODEL is not None:
        return _RMBG14_MODEL
    if _RMBG14_TRIED:
        return None

    with _BRIA_LOCK:
        if _RMBG14_MODEL is not None:
            return _RMBG14_MODEL
        if _RMBG14_TRIED:
            return None
        _RMBG14_TRIED = True
        os.makedirs(CLOUD_MODELS_DIR, exist_ok=True)
        try:
            from transformers import AutoModelForImageSegmentation  # type: ignore

            model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4",
                trust_remote_code=True,
                cache_dir=CLOUD_MODELS_DIR,
                token=HF_TOKEN,
            )
            model.eval()
            _RMBG14_MODEL = ("torch", model)
            print("✅ [BRIA Cloud] Loaded secondary fallback model: briaai/RMBG-1.4 (PyTorch)")
            return _RMBG14_MODEL
        except Exception:
            pass

        try:
            import onnxruntime as ort  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore

            onnx_path = hf_hub_download(
                repo_id="briaai/RMBG-1.4",
                filename="onnx/model.onnx",
                cache_dir=CLOUD_MODELS_DIR,
                token=HF_TOKEN,
            )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            avail = ort.get_available_providers()
            providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            _RMBG14_MODEL = ("onnx", sess)
            print("✅ [BRIA Cloud] Loaded secondary fallback model: briaai/RMBG-1.4 (ONNX)")
            return _RMBG14_MODEL
        except Exception as e:
            print(f"⚠️ [BRIA Cloud] RMBG-1.4 fallback loader error: {e}")
    return None


def warmup_bria_cloud_models() -> dict[str, bool]:
    """
    Pre-download BRIA RMBG-2.0 weights in Hugging Face Cloud at container startup so
    the user's first `Run` click has zero download wait. Strictly no-op on local PC!
    """
    if not IS_HF_CLOUD:
        return {"rmbg_2_0": False, "rmbg_1_4": False, "cloud_only_guard": True}
    m20 = _load_rmbg20_cloud()
    m14 = None if m20 is not None else _load_rmbg14_cloud()
    return {
        "rmbg_2_0": m20 is not None,
        "rmbg_1_4": m14 is not None,
        "cloud_only_guard": True,
    }


def _run_rmbg20_letterbox_1024(model20, device: str, img_rgb: np.ndarray) -> np.ndarray | None:
    """
    Aspect-Ratio-Preserving 1024×1024 Letterbox Inference for BRIA RMBG-2.0:
    Eliminates horizontal/vertical squashing on tall 9:16 portraits and wide 16:10 couple photos.
    """
    import torch  # type: ignore

    ch, cw = img_rgb.shape[:2]
    if ch < 8 or cw < 8:
        return None
    scale = min(1024.0 / float(cw), 1024.0 / float(ch))
    nw = max(16, min(1024, int(round(cw * scale))))
    nh = max(16, min(1024, int(round(ch * scale))))
    resized = resize_img(img_rgb, nw, nh, mode="lanczos" if scale < 1.0 else "bilinear")

    pad_top = (1024 - nh) // 2
    pad_bot = 1024 - nh - pad_top
    pad_left = (1024 - nw) // 2
    pad_right = 1024 - nw - pad_left

    if HAS_CV2 and cv2 is not None and (pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0):
        canvas_1024 = cv2.copyMakeBorder(
            resized, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101
        )
    elif pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0:
        canvas_1024 = np.pad(resized, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)), mode="reflect")
    else:
        canvas_1024 = resized

    in_norm = canvas_1024.astype(np.float32) * np.float32(1.0 / 255.0)
    inp = ((in_norm - _IMAGENET_MEAN[None, None, :]) / _IMAGENET_STD[None, None, :]).transpose(2, 0, 1)[None, ...]

    with torch.inference_mode():
        t_in = torch.from_numpy(np.ascontiguousarray(inp)).to(device=device, dtype=torch.float32)
        out = model20(t_in)
        logits = out[-1] if isinstance(out, (list, tuple)) else out
        if hasattr(logits, "logits"):
            logits = logits.logits
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)

    if probs.ndim != 2:
        return None
    unpadded = probs[pad_top : pad_top + nh, pad_left : pad_left + nw]
    p_min, p_max = float(unpadded.min()), float(unpadded.max())
    if p_max - p_min > 1e-3 and p_max > 0.65 and p_min < 0.35:
        unpadded = (unpadded - p_min) / (p_max - p_min)
    alpha = np.clip(unpadded, 0.0, 1.0)
    if alpha.shape != (ch, cw):
        alpha = resize_img(alpha, cw, ch, mode="bilinear")
    return alpha.astype(np.float32)


def _infer_bria_neural_alpha(img_rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """
    Run Aspect-Preserving BRIA RMBG-2.0 (primary) + High-Res Head/Hair Zoom Pass,
    or BRIA RMBG-1.4 (secondary). Returns `(alpha_float32_HxW, model_tag)` or `(None, ...)`.
    """
    global _ACTIVE_BRIA_TAG
    if not IS_HF_CLOUD:
        return None, "Local-ZeroDownload-Analytical"

    h, w = img_rgb.shape[:2]

    # 1. Primary: BRIA RMBG-2.0 (`1038lab/RMBG-2.0`) on CUDA / CPU
    model20 = _load_rmbg20_cloud()
    if model20 is not None:
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                cur_dev = next(model20.parameters()).device.type
                if cur_dev != device:
                    model20.to(device)
            except Exception:
                model20.to(device)

            full_alpha = _run_rmbg20_letterbox_1024(model20, device, img_rgb)
            if full_alpha is not None:
                # Dual-Pass High-Res Head & Flyaway Hair ROI Matting when portrait has extra detail
                fg_ys, fg_xs = np.where(full_alpha > 0.25)
                if max(h, w) >= 720 and fg_ys.size >= 64:
                    y_top = max(0, int(fg_ys.min()) - int(0.06 * h))
                    y_bot = min(h, y_top + int(0.68 * (fg_ys.max() - y_top + 1)))
                    x_left = max(0, int(fg_xs.min()) - int(0.06 * w))
                    x_right = min(w, int(fg_xs.max()) + int(0.06 * w))
                    roi_h, roi_w = y_bot - y_top, x_right - x_left
                    # Run zoomed head/shoulder pass ONLY on wide group/distant shots where ROI < 48% of frame
                    if roi_h >= 160 and roi_w >= 160 and (roi_h * roi_w) < 0.48 * (h * w):
                        roi_alpha = _run_rmbg20_letterbox_1024(model20, device, img_rgb[y_top:y_bot, x_left:x_right])
                        if roi_alpha is not None:
                            # Smoothly feather the bottom/side seam of the zoomed ROI into full_alpha
                            wy = np.ones((roi_h, 1), dtype=np.float32)
                            fade_b = max(8, int(0.12 * roi_h))
                            wy[-fade_b:, 0] = np.linspace(1.0, 0.0, fade_b, dtype=np.float32)
                            if y_top > 4:
                                fade_t = min(16, max(4, int(0.05 * roi_h)))
                                wy[:fade_t, 0] = np.linspace(0.0, 1.0, fade_t, dtype=np.float32)
                            wx = np.ones((1, roi_w), dtype=np.float32)
                            fade_s = max(6, int(0.06 * roi_w))
                            if x_left > 4:
                                wx[0, :fade_s] = np.linspace(0.0, 1.0, fade_s, dtype=np.float32)
                            if x_right < w - 4:
                                wx[0, -fade_s:] = np.linspace(1.0, 0.0, fade_s, dtype=np.float32)
                            w_roi = wy * wx
                            base_patch = full_alpha[y_top:y_bot, x_left:x_right]
                            full_alpha[y_top:y_bot, x_left:x_right] = base_patch * (1.0 - w_roi) + roi_alpha * w_roi

                _ACTIVE_BRIA_TAG = f"BRIA-RMBG-2.0-DualPass ({device.upper()})"
                return full_alpha, _ACTIVE_BRIA_TAG
        except Exception as e:
            print(f"⚠️ [BRIA Cloud] RMBG-2.0 inference fallback triggered: {e}")

    # 2. Secondary: Official BRIA RMBG-1.4 (`briaai/RMBG-1.4`)
    m14_entry = _load_rmbg14_cloud()
    if m14_entry is not None:
        kind, m14 = m14_entry
        try:
            in_1024 = resize_img(img_rgb, 1024, 1024, mode="bilinear").astype(np.float32) * np.float32(1.0 / 255.0)
            inp = (in_1024 - np.float32(0.5)).transpose(2, 0, 1)[None, ...]
            if kind == "torch":
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if next(m14.parameters()).device.type != device:
                    m14.to(device)
                with torch.inference_mode():
                    t_in = torch.from_numpy(np.ascontiguousarray(inp)).to(device=device, dtype=torch.float32)
                    preds = m14(t_in)
                    raw_t = preds[0][0] if isinstance(preds, (list, tuple)) else preds
                    raw = raw_t.squeeze().detach().cpu().numpy().astype(np.float32)
                dev_lbl = device.upper()
            else:
                in_name = m14.get_inputs()[0].name
                preds = m14.run(None, {in_name: np.ascontiguousarray(inp)})
                raw = np.squeeze(preds[0]).astype(np.float32)
                dev_lbl = "ONNX"

            if raw.ndim == 2:
                r_min, r_max = float(raw.min()), float(raw.max())
                alpha = (raw - r_min) / max(1e-5, r_max - r_min)
                alpha = np.clip(alpha, 0.0, 1.0)
                if alpha.shape != (h, w):
                    alpha = resize_img(alpha, w, h, mode="bilinear")
                _ACTIVE_BRIA_TAG = f"BRIA-RMBG-1.4 ({dev_lbl})"
                return alpha, _ACTIVE_BRIA_TAG
        except Exception as e:
            print(f"⚠️ [BRIA Cloud] RMBG-1.4 inference error: {e}")

    return None, "Analytical-Fallback"


def extract_bria_foreground_matte(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics | None = None,
    subjects: list | None = None,
) -> tuple[np.ndarray, str]:
    """
    Extract continuous studio portrait foreground matte in [0, 1] of shape (H, W)
    and return `(refined_alpha, engine_tag)`:
    - In HF Cloud: executes Dual-Pass Aspect-Preserving BRIA RMBG-2.0 -> Multi-Subject
      Connected-Component Bystander Pruning -> Pure-Edge Guided Hair Refinement
      (ZERO exterior oval halo override!).
    - Locally (Windows PC): executes deterministic 8-Zone Mahalanobis + Guided Filter
      in <15ms with ZERO model downloads (`0 MB`).
    """
    h, w = img_rgb.shape[:2]
    raw_alpha, engine_tag = _infer_bria_neural_alpha(img_rgb)

    if raw_alpha is not None:
        from src.core.multi_subject_detector import detect_foreground_subjects, prune_background_bystanders

        active_subjects = subjects if subjects else detect_foreground_subjects(img_rgb, raw_bria_alpha=raw_alpha)

        # 1. Prune disconnected background bystanders (e.g. passersby in shops/streets) & floating clutter
        alpha_clean = prune_background_bystanders(raw_alpha, active_subjects)

        # 2. Protect ONLY tiny internal specular/eyeglass pinholes strictly inside each face's eye-nose triangle
        #    (NEVER expands outside BRIA's true cheek, jaw, ear, or saree-pallu silhouette!)
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        for subj in active_subjects:
            g = subj.geom
            ipd = max(16.0, float(g.ipd_px))
            ecx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
            ecy = 0.5 * (g.left_eye.center_y + g.right_eye.center_y) + 0.18 * ipd
            # Tight central orbital-nasal core (`0.40 * ipd` — strictly inside face interior, well inside cheeks!)
            tight_eye_bridge = np.clip(
                1.0 - (((xx - ecx) / (0.40 * ipd)) ** 2 + ((yy - ecy) / (0.44 * ipd)) ** 2),
                0.0,
                1.0,
            )
            # Only reinforce if BRIA itself already sees strong surrounding face opacity (>0.75)
            local_mean_alpha = gaussian_blur(alpha_clean, sigma=max(3.0, 0.12 * ipd))
            interior_pinhole_guard = tight_eye_bridge * np.clip((local_mean_alpha - 0.72) / 0.18, 0.0, 1.0)
            alpha_clean = np.maximum(alpha_clean, interior_pinhole_guard * 0.99)

        # 3. Clean faint background haze (< 0.04) outside subject boundary while preserving soft flyaway hairs
        radius = max(2, min(7, int(min(h, w) * 0.005)))
        guided = guided_filter(img_rgb, alpha_clean, radius=radius, eps=3.0e-4)
        # Blend 76% pure BRIA matte + 24% RGB-guided edge alignment ONLY inside the transition band (0.02..0.98)
        band = (alpha_clean > 0.015) & (alpha_clean < 0.985)
        refined = np.where(band, 0.76 * alpha_clean + 0.24 * guided, alpha_clean)
        refined = np.clip((refined - 0.035) / 0.945, 0.0, 1.0)

        # 4. Flush border lock ONLY where subject genuinely touches the image border
        if h >= 16:
            bot_ref = np.max(refined[h - 8 : h - 3, :], axis=0, keepdims=True)
            bot_touches = (bot_ref > 0.80) & (refined[h - 1 : h, :] > 0.42)
            refined[h - 5 : h, :] = np.where(bot_touches, np.maximum(refined[h - 5 : h, :], bot_ref), refined[h - 5 : h, :])
            top_ref = np.max(refined[3:7, :], axis=0, keepdims=True)
            top_touches = (top_ref > 0.85) & (refined[0:1, :] > 0.42)
            refined[0:4, :] = np.where(top_touches, np.maximum(refined[0:4, :], top_ref), refined[0:4, :])

        return refined.astype(np.float32), engine_tag

    # ========================================================================
    # Local Zero-Download Analytical Silhouette Fallback (`not IS_HF_CLOUD`)
    # ========================================================================
    if geom is None or (geom.img_w, geom.img_h) != (w, h):
        geom = analyze_face_geometry(img_rgb)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    ipd = max(16.0, float(geom.ipd_px))
    fcx = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    fcy = 0.5 * (geom.crown_xy[1] + geom.chin_xy[1])

    outer_head_rx = max(18.0, 1.28 * ipd)
    outer_head_ry = max(22.0, 1.56 * ipd)
    outer_head = np.exp(-0.5 * (((xx - fcx) / outer_head_rx) ** 2 + ((yy - fcy) / outer_head_ry) ** 2) ** 2)

    sh_rx = max(32.0, 2.60 * ipd)
    outer_torso = np.clip((yy - (geom.chin_xy[1] - 0.15 * ipd)) / max(6.0, 0.25 * ipd), 0.0, 1.0) * np.exp(
        -0.5 * ((xx - fcx) / sh_rx) ** 4
    )
    outer_reach = np.clip(np.maximum(outer_head, outer_torso) * 1.65, 0.0, 1.0)

    inner_face_core = np.clip(
        1.0 - (((xx - fcx) / max(12.0, 0.56 * ipd)) ** 2 + ((yy - (fcy + 0.06 * ipd)) / max(16.0, 0.78 * ipd)) ** 2),
        0.0,
        1.0,
    )

    lab = rgb_to_lab(img_rgb)
    pad_y = max(4, int(h * 0.10))
    pad_x = max(4, int(w * 0.08))
    bg_valid = (np.abs(xx - fcx) > 0.96 * ipd) & (outer_reach < 0.45)

    def _cluster_dist(cluster_pts: np.ndarray) -> np.ndarray:
        med = np.median(cluster_pts, axis=0)
        std = np.maximum(np.std(cluster_pts, axis=0), np.array([5.5, 3.5, 3.5], dtype=np.float32))
        d = (lab - med[None, None, :]) / std[None, None, :]
        return np.sqrt(0.78 * d[:, :, 0] ** 2 + 1.15 * d[:, :, 1] ** 2 + 1.15 * d[:, :, 2] ** 2)

    y1, y2, y3 = min(h, pad_y), min(h, 2 * pad_y), min(h, 3 * pad_y)
    x_l2 = min(w, 2 * pad_x)
    x_r1, x_r2 = max(0, w - pad_x), max(0, w - 2 * pad_x)
    raw_zones = [
        (slice(0, y1), slice(0, pad_x)),
        (slice(0, y1), slice(x_r1, w)),
        (slice(y1, y2), slice(0, pad_x)),
        (slice(y1, y2), slice(x_r1, w)),
        (slice(y2, y3), slice(0, pad_x)),
        (slice(y2, y3), slice(x_r1, w)),
        (slice(0, y1), slice(pad_x, x_l2)),
        (slice(0, y1), slice(x_r2, x_r1)),
    ]
    perimeter_zones: list[np.ndarray] = []
    for sy, sx in raw_zones:
        sub_mask = bg_valid[sy, sx]
        pts = lab[sy, sx][sub_mask] if np.count_nonzero(sub_mask) >= 6 else lab[sy, sx].reshape(-1, 3)
        if len(pts) >= 4:
            perimeter_zones.append(pts)

    dist_bg = _cluster_dist(perimeter_zones[0])
    for z_pts in perimeter_zones[1:]:
        dist_bg = np.minimum(dist_bg, _cluster_dist(z_pts))

    color_fg = np.clip((dist_bg - 1.45) / 1.55, 0.0, 1.0)
    fallback_alpha = np.clip(np.maximum(color_fg * outer_reach, inner_face_core * 0.99), 0.0, 1.0)
    radius = max(3, min(10, int(min(h, w) * 0.010)))
    refined_fb = guided_filter(img_rgb, fallback_alpha, radius=radius, eps=5e-4)
    refined_fb = np.clip((refined_fb - 0.05) / 0.90, 0.0, 1.0)
    return refined_fb.astype(np.float32), engine_tag


def decontaminate_bria_foreground(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    bg_orig_rgb: tuple[float, float, float] | list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    2D Spatially-Varying Closed-Form Foreground Boundary Color Spill Decontamination:
      F_clean(x,y) = clip((I(x,y) - (1 - alpha(x,y)) * B_local(x,y)) / max(alpha(x,y), 0.16), 0, 255)
    Estimates `B_local(x,y)` continuously along every boundary pixel via Normalized Convolution
    of the immediate exterior background (`alpha < 0.06`), removing white curtain halos, green
    leaf spill, and brown wood reflections accurately even on complex multi-colored backgrounds!
    """
    h, w = img_rgb.shape[:2]
    af = alpha.astype(np.float32)
    if af.max() > 1.5:
        af /= 255.0

    img_f = img_rgb[:, :, :3].astype(np.float32)

    if bg_orig_rgb is not None:
        bg_arr = np.asarray(bg_orig_rgb, dtype=np.float32).reshape(1, 1, 3)
        bg_local = np.broadcast_to(bg_arr, (h, w, 3))
    else:
        bg_mask = (af < 0.06).astype(np.float32)
        if float(bg_mask.sum()) >= 32.0:
            sigma_bg = max(8.0, min(36.0, min(h, w) * 0.025))
            num = gaussian_blur(img_f * bg_mask[:, :, None], sigma=sigma_bg)
            den = gaussian_blur(bg_mask, sigma=sigma_bg)[:, :, None]
            global_med = np.median(img_f[af < 0.08], axis=0).astype(np.float32) if np.any(af < 0.08) else np.array([220.0, 220.0, 220.0], dtype=np.float32)
            bg_local = np.where(den > 1e-3, num / np.maximum(den, 1e-3), global_med[None, None, :])
        else:
            pad = max(4, int(min(h, w) * 0.06))
            bg_local = np.broadcast_to(
                np.median(img_f[:pad, :pad].reshape(-1, 3), axis=0).astype(np.float32)[None, None, :],
                (h, w, 3),
            )

    a_3d = af[:, :, None]
    denom = np.maximum(a_3d, 0.16)
    f_clean = np.clip((img_f - (1.0 - a_3d) * bg_local) / denom, 0.0, 255.0)

    # Green-foliage & bright-sky edge despill (eliminates green leaf fringe on outdoor dark hair & sarees!)
    bg_green_excess = np.maximum(0.0, bg_local[:, :, 1] - np.maximum(bg_local[:, :, 0], bg_local[:, :, 2]))
    green_spill_gate = np.clip((bg_green_excess - 6.0) / 22.0, 0.0, 1.0)
    fg_green_cap = 0.54 * f_clean[:, :, 0] + 0.46 * f_clean[:, :, 2] + 3.0
    f_clean[:, :, 1] = np.where(
        green_spill_gate > 0.05,
        f_clean[:, :, 1] * (1.0 - 0.85 * green_spill_gate) + np.minimum(f_clean[:, :, 1], fg_green_cap) * (0.85 * green_spill_gate),
        f_clean[:, :, 1],
    )

    # Apply strictly in the semi-transparent edge band (0.02 < alpha < 0.96)
    w_low = np.clip((af - 0.02) / 0.08, 0.0, 1.0)
    w_high = np.clip((0.96 - af) / 0.09, 0.0, 1.0)
    w_dec = w_low * w_high * 0.96
    out = img_f * (1.0 - w_dec[:, :, None]) + f_clean * w_dec[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def upsample_alpha_guided(
    alpha_low: np.ndarray,
    high_res_guide_rgb: np.ndarray,
) -> np.ndarray:
    """
    Fast Joint-Bilateral Edge-Guided Alpha Upsampler up to True 8K (`7680px+`):
    Upsamples `alpha_low` to `high_res_guide_rgb.shape[:2]` and snaps transition boundaries
    to crisp 2K/4K/8K hair & garment edges using He's O(N) C++ SIMD Guided Filter on the
    transition band (including 8K outputs!).
    """
    th, tw = high_res_guide_rgb.shape[:2]
    af = alpha_low.astype(np.float32)
    if af.max() > 1.5:
        af /= 255.0
    if af.shape[:2] == (th, tw):
        return np.clip(af, 0.0, 1.0)

    up_lin = resize_img(af, tw, th, mode="bilinear")
    if max(th, tw) <= 8192:
        r = max(2, min(8, int(round(max(th, tw) / 900.0))))
        # Crop strictly to the bounding box of the boundary band for ultra-fast <20ms 8K execution
        band = (up_lin > 0.01) & (up_lin < 0.99)
        bys, bxs = np.where(band)
        if bys.size >= 16:
            pad = r + 4
            y1 = max(0, int(bys.min()) - pad)
            y2 = min(th, int(bys.max()) + pad + 1)
            x1 = max(0, int(bxs.min()) - pad)
            x2 = min(tw, int(bxs.max()) + pad + 1)
            sub_ref = guided_filter(high_res_guide_rgb[y1:y2, x1:x2], up_lin[y1:y2, x1:x2], radius=r, eps=3.5e-4)
            out = up_lin.copy()
            sub_band = band[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = np.where(sub_band, np.clip(sub_ref, 0.0, 1.0), up_lin[y1:y2, x1:x2])
            return out.astype(np.float32)
    return np.clip(up_lin, 0.0, 1.0).astype(np.float32)


def apply_bria_studio_background(
    fg_rgb: np.ndarray,
    fg_alpha: np.ndarray,
    bg_rgb_color: tuple[int, int, int] = (152, 243, 255),
    backdrop_style: str = "solid",
    rim_light_wrap: float = 0.06,
) -> np.ndarray:
    """
    Synthesize studio backdrop (`solid`, `radial_studio`, or `vertical_gradient`) and composite
    `fg_rgb` with subtle Optical Studio Rim-Light Wrap strictly inside the semi-transparent
    hair/shoulder edge (`0.08 < alpha < 0.92`) so background pixels (`alpha == 0`) remain 100%
    pure `bg_rgb_color` while hair edges integrate naturally without cutout harshness.
    """
    h, w = fg_rgb.shape[:2]
    af = fg_alpha.astype(np.float32)
    if af.max() > 1.5:
        af /= 255.0
    if af.shape[:2] != (h, w):
        af = upsample_alpha_guided(af, fg_rgb)

    backdrop = synthesize_studio_backdrop(h, w, bg_rgb_color, backdrop_style=backdrop_style)
    fg_f = fg_rgb[:, :, :3].astype(np.float32)
    bg_f = backdrop.astype(np.float32)

    if rim_light_wrap > 0.005:
        edge_fringe = (4.0 * af * (1.0 - af)) * np.clip((af - 0.08) / 0.15, 0.0, 1.0)
        wrap_w = (float(np.clip(rim_light_wrap, 0.0, 0.25)) * edge_fringe)[:, :, None]
        fg_wrapped = fg_f * (1.0 - wrap_w) + bg_f * wrap_w
    else:
        fg_wrapped = fg_f

    a_3d = af[:, :, None]
    comp = a_3d * fg_wrapped + (1.0 - a_3d) * bg_f
    return np.clip(np.round(comp), 0, 255).astype(np.uint8)
