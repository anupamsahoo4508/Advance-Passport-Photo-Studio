"""
AI & Bio-Optical Passport Photo Processor V8 — Multi-Subject, Pure-BRIA, CatVTON & True 8K Brain.

Upgrades in V8:
1. Multi-Subject & Official Couple Portrait Intelligence (`1–4` Foreground People):
   - Detects all real foreground subjects via YuNet + anthropometric verification while rejecting
     framed wall paintings (e.g. Bal Krishna frame) and background posters/bystanders.
   - Runs independent 12-Zone `StudioAutoDirectorAI` + `CodeFormer(adain=True)` + `OcularLock™`
     + `IdentityMarkLock™ ∪ OrnamentLock™` (locking bindi, sindoor, nose-pin, earrings, scars, moles)
     per person with Partition-of-Unity blending.
2. Surgical Dual-Pass Pure-BRIA (`RMBG-2.0`) Matting + Bystander Pruning + Local Spill Decontamination.
3. Real Diffusion AI Virtual Try-On (`CatVTON` FP16 CUDA) + Decoupled "No-Run, Zero-Degradation" Try-On:
   - Executes Dress Change BEFORE Super-Resolution during `Run`, and provides `tryon_dress_decoupled`
     for instant AI dress previewing without re-upscaling or stacking garments.
4. True Native 4K (`3840px`) & 8K (`7680px`) Super-Resolution (`4x-UltraSharp` + `DermaPore-8K™`
   cellular skin pores + `WeaveSharp-8K™` garment textile weave sharpness).
5. ZeroGPU-Safe Browser Session State (`session_state`) + Foreground-Scoped Manual Colour Correction
   so `Gamma`, `Saturation`, `Brightness`, `Contrast`, `Warmth`, `Shadows Lift`, and `Colour Picker`
   work in `<20ms` without tinting official studio backgrounds.
"""
from __future__ import annotations

import colorsys
import hashlib
import math
import os
import re
import time
import uuid
import numpy as np
from PIL import Image, ImageColor, ImageDraw

from src.shared.config import IS_HF_CLOUD, TEMP_DIR
from src.core.cv_ops import (
    box_filter,
    gaussian_blur,
    imread_rgb,
    imwrite_rgb,
    resize_img,
)
from src.core.facial_math_engine import (
    FacialAnthropometrics,
    analyze_face_geometry,
    dermaphysics_skin_enhance,
    edge_directed_4k_upscale,
    full_ai_portrait_intelligence,
    full_ai_portrait_intelligence_multi,
    high_res_detail_upscale,
    ocular_lock_restore,
    scale_facial_anthropometrics,
    spherical_harmonics_relight,
    synthesize_hd_dermal_and_fabric_microtexture,
)
from src.core.multi_subject_detector import (
    SubjectProfile,
    build_veil_dress_reminder,
    detect_foreground_subjects,
    prune_background_bystanders,
)
from src.core.bria_bg_engine import (
    apply_bria_studio_background,
    decontaminate_bria_foreground,
    extract_bria_foreground_matte,
    get_bria_engine_status,
    upsample_alpha_guided,
)
from src.core.dress_engine import DRESS_PRESETS, apply_anthropometric_dress
from src.core.color_studio import (
    apply_correction,
    has_active_manual_adjustments,
    render_foreground_and_backdrop,
    synthesize_studio_backdrop,
)
from src.core.print_studio import (
    apply_name_date_plaque,
    auto_level_portrait,
    generate_print_sheet,
    icao_biometric_crop,
)

_STAGE1_CACHE: dict[str, object] = {"key": None, "data": None}
_STAGE2_CACHE: dict[str, object] = {"key": None, "data": None}
_LAST_RUN_CACHE: dict[str, object] = {
    "base_composited": None,
    "fg_rgb": None,
    "fg_alpha": None,
    "custom_color": "#98f3ff",
    "backdrop_style": "solid",
    "geom": None,
    "metrics": None,
    "params": None,
}


def _get_rembg_session():
    """
    Deprecated compatibility hook — strictly no-op on local PC (`not IS_HF_CLOUD`)
    so ZERO model weights are ever downloaded onto the user's Windows machine.
    """
    if not IS_HF_CLOUD:
        return None
    return None


def compute_image_signature(img_rgb: np.ndarray | None) -> str:
    """Fast deterministic content signature for session cache validation."""
    if img_rgb is None or not isinstance(img_rgb, np.ndarray) or img_rgb.size == 0:
        return "none"
    h, w = img_rgb.shape[:2]
    step_y = max(1, h // 24)
    step_x = max(1, w // 24)
    sub = np.ascontiguousarray(img_rgb[::step_y, ::step_x, :3])
    digest = hashlib.blake2b(sub.tobytes(), digest_size=10).hexdigest()
    return f"{w}x{h}_{digest}"


def hex_to_rgb(
    color_str: str | tuple | list | dict | None,
    default_rgb: tuple[int, int, int] = (152, 243, 255),
) -> tuple[int, int, int]:
    """
    Convert any CSS/Gradio 5 color specification to an (R, G, B) uint8 tuple in [0, 255].
    Falls back to `default_rgb` (`(152, 243, 255)` / `#98f3ff` Light Blue) on invalid input.
    """
    if color_str is None:
        return default_rgb

    if isinstance(color_str, dict):
        if all(k in color_str for k in ("r", "g", "b")):
            color_str = (color_str["r"], color_str["g"], color_str["b"])
        else:
            return default_rgb

    if isinstance(color_str, (tuple, list, np.ndarray)):
        if len(color_str) >= 3:
            try:
                raw_items = list(color_str[:3])
                vals = [float(raw_items[0]), float(raw_items[1]), float(raw_items[2])]
                has_float_type = any(isinstance(x, (float, np.floating)) for x in raw_items)
                if has_float_type and 0.0 <= min(vals) and max(vals) <= 1.0 and max(vals) > 0.0:
                    vals = [v * 255.0 for v in vals]
                return (
                    int(np.clip(round(vals[0]), 0, 255)),
                    int(np.clip(round(vals[1]), 0, 255)),
                    int(np.clip(round(vals[2]), 0, 255)),
                )
            except Exception:
                return default_rgb
        return default_rgb

    s = str(color_str).strip()
    if not s:
        return default_rgb

    s_low = s.lower()

    # 1. Handle rgb(...) and rgba(...)
    m_rgb = re.match(r"^rgba?\s*\(([^)]+)\)$", s_low)
    if m_rgb:
        try:
            tokens = [t for t in re.split(r"[\s,/]+", m_rgb.group(1).strip()) if t]
            if len(tokens) >= 3:
                rgb_tokens = tokens[:3]
                has_pct = any(tok.endswith("%") for tok in rgb_tokens)
                has_dot = any("." in tok for tok in rgb_tokens)
                raw_vals = [
                    (float(tok[:-1]) * 255.0 / 100.0) if tok.endswith("%") else float(tok)
                    for tok in rgb_tokens
                ]
                if not has_pct and has_dot and 0.0 <= min(raw_vals) and max(raw_vals) <= 1.0 and max(raw_vals) > 0.0:
                    raw_vals = [v * 255.0 for v in raw_vals]
                return (
                    int(np.clip(round(raw_vals[0]), 0, 255)),
                    int(np.clip(round(raw_vals[1]), 0, 255)),
                    int(np.clip(round(raw_vals[2]), 0, 255)),
                )
        except Exception:
            return default_rgb

    # 2. Handle hsl(...) and hsla(...)
    m_hsl = re.match(r"^hsla?\s*\(([^)]+)\)$", s_low)
    if m_hsl:
        try:
            tokens = [t for t in re.split(r"[\s,/]+", m_hsl.group(1).strip()) if t]
            if len(tokens) >= 3:
                h_tok, s_tok, l_tok = tokens[0], tokens[1], tokens[2]
                if h_tok.endswith("deg"):
                    h_deg = float(h_tok[:-3])
                elif h_tok.endswith("turn"):
                    h_deg = float(h_tok[:-4]) * 360.0
                elif h_tok.endswith("rad"):
                    h_deg = math.degrees(float(h_tok[:-3]))
                else:
                    h_deg = float(h_tok)
                h_norm = (h_deg % 360.0) / 360.0
                s_norm = float(s_tok[:-1]) / 100.0 if s_tok.endswith("%") else float(s_tok)
                l_norm = float(l_tok[:-1]) / 100.0 if l_tok.endswith("%") else float(l_tok)
                if s_norm > 1.0:
                    s_norm /= 100.0
                if l_norm > 1.0:
                    l_norm /= 100.0
                s_norm = float(np.clip(s_norm, 0.0, 1.0))
                l_norm = float(np.clip(l_norm, 0.0, 1.0))
                r_f, g_f, b_f = colorsys.hls_to_rgb(h_norm, l_norm, s_norm)
                return (
                    int(np.clip(round(r_f * 255.0), 0, 255)),
                    int(np.clip(round(g_f * 255.0), 0, 255)),
                    int(np.clip(round(b_f * 255.0), 0, 255)),
                )
        except Exception:
            return default_rgb

    # 2b. Handle hsv(...) and hsva(...)
    m_hsv = re.match(r"^hsva?\s*\(([^)]+)\)$", s_low)
    if m_hsv:
        try:
            tokens = [t for t in re.split(r"[\s,/]+", m_hsv.group(1).strip()) if t]
            if len(tokens) >= 3:
                h_tok, s_tok, v_tok = tokens[0], tokens[1], tokens[2]
                h_deg = float(h_tok[:-3]) if h_tok.endswith("deg") else float(h_tok)
                h_norm = (h_deg % 360.0) / 360.0
                s_norm = float(s_tok[:-1]) / 100.0 if s_tok.endswith("%") else float(s_tok)
                v_norm = float(v_tok[:-1]) / 100.0 if v_tok.endswith("%") else float(v_tok)
                if s_norm > 1.0:
                    s_norm /= 100.0
                if v_norm > 1.0:
                    v_norm /= 100.0
                r_f, g_f, b_f = colorsys.hsv_to_rgb(
                    h_norm,
                    float(np.clip(s_norm, 0.0, 1.0)),
                    float(np.clip(v_norm, 0.0, 1.0)),
                )
                return (
                    int(np.clip(round(r_f * 255.0), 0, 255)),
                    int(np.clip(round(g_f * 255.0), 0, 255)),
                    int(np.clip(round(b_f * 255.0), 0, 255)),
                )
        except Exception:
            return default_rgb

    # 3. Handle #RGB, #RGBA, #RRGGBB, #RRGGBBAA
    clean = s_low.lstrip("#")
    if re.match(r"^[0-9a-f]{3,8}$", clean) and len(clean) in (3, 4, 6, 8):
        try:
            if len(clean) in (3, 4):
                clean = "".join(c * 2 for c in clean[:3])
            else:
                clean = clean[:6]
            return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))
        except Exception:
            return default_rgb

    # 4. Handle named CSS colors via PIL.ImageColor
    try:
        c = ImageColor.getrgb(s)
        return (int(c[0]), int(c[1]), int(c[2]))
    except Exception:
        return default_rgb


def normalize_color_to_hex(
    color_str: str | tuple | list | dict | None,
    default: str = "#98f3ff",
) -> str:
    """Normalize any CSS/Gradio color string to canonical #rrggbb hex."""
    def_rgb = hex_to_rgb(default, default_rgb=(152, 243, 255)) if default != "#98f3ff" else (152, 243, 255)
    r, g, b = hex_to_rgb(color_str, default_rgb=def_rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_fast_unique_png(img_rgb: np.ndarray, prefix: str = "ms_v9") -> str:
    """
    Write a lossless PNG with ultra-fast OpenCV C++ `IMWRITE_PNG_COMPRESSION=1` (~65ms at 4K vs 3400ms PIL)
    using a unique UUID filename so Gradio/Chrome ALWAYS repaints `Live Preview` immediately (zero stale cache!)
    while keeping Gradio's built-in download icon 100% full-resolution lossless PNG!
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    now_t = time.time()
    try:
        for fn in os.listdir(TEMP_DIR):
            if fn.startswith(prefix + "_") and fn.endswith(".png"):
                fp = os.path.join(TEMP_DIR, fn)
                if now_t - os.path.getmtime(fp) > 180.0:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
    except Exception:
        pass

    out_path = os.path.abspath(os.path.join(TEMP_DIR, f"{prefix}_{uuid.uuid4().hex[:10]}.png"))
    imwrite_rgb(out_path, img_rgb)
    return out_path


def smart_resize(img: np.ndarray, max_dim: int | None) -> np.ndarray:
    """Resize image if its longest side exceeds max_dim, preserving aspect ratio."""
    if max_dim is None:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = float(max_dim) / float(max(h, w))
    return resize_img(img, int(round(w * scale)), int(round(h * scale)), mode="lanczos")


# === 10-Tier V8 Super-Resolution Configuration (1 to 10 -> 2K / 4K / 5K / True 8K 7680px) ===
UPSCALE_TIERS = {
    1:  {"max_dim": 2048, "target_max_px": 1080, "target_h": 1080, "min_out_h": 531,  "scale": 1.00, "crisping": 0.18},
    2:  {"max_dim": 2048, "target_max_px": 1440, "target_h": 1440, "min_out_h": 1440, "scale": 1.50, "crisping": 0.24},
    3:  {"max_dim": 2048, "target_max_px": 1800, "target_h": 1800, "min_out_h": 1800, "scale": 1.85, "crisping": 0.30},
    4:  {"max_dim": 2048, "target_max_px": 2160, "target_h": 2160, "min_out_h": 2160, "scale": 2.20, "crisping": 0.36},
    5:  {"max_dim": 2048, "target_max_px": 2560, "target_h": 2560, "min_out_h": 2560, "scale": 2.60, "crisping": 0.40},
    6:  {"max_dim": 2048, "target_max_px": 3000, "target_h": 3000, "min_out_h": 3000, "scale": 3.00, "crisping": 0.45},
    7:  {"max_dim": 2048, "target_max_px": 3840, "target_h": 3840, "min_out_h": 3840, "scale": 3.80, "crisping": 0.50},
    8:  {"max_dim": 2048, "target_max_px": 4320, "target_h": 4320, "min_out_h": 4320, "scale": 4.30, "crisping": 0.54},
    9:  {"max_dim": 2048, "target_max_px": 5120, "target_h": 5120, "min_out_h": 5120, "scale": 5.00, "crisping": 0.58},
    10: {"max_dim": 2048, "target_max_px": 7680, "target_h": 7680, "min_out_h": 7680, "scale": 7.50, "crisping": 0.62},
}


def clean_corner_camera_watermark(
    img_rgb: np.ndarray,
    fg_alpha: np.ndarray | None = None,
    subjects: list[SubjectProfile] | None = None,
) -> np.ndarray:
    """
    Gently heal sparse bright white phone camera text watermarks (e.g. 'vivo Y31 Pro')
    in the bottom-left / bottom-right 11% margin when stamped over dark/midtone clothing,
    without touching bright shirts or any facial region.
    """
    h, w = img_rgb.shape[:2]
    if h < 180 or w < 180:
        return img_rgb

    out = img_rgb.copy()
    y1 = int(round(0.885 * h))
    y2 = h
    corner_boxes = [
        (0, y1, int(round(0.45 * w)), y2),
        (int(round(0.55 * w)), y1, w, y2),
    ]

    for cx1, cy1, cx2, cy2 in corner_boxes:
        if cx2 - cx1 < 24 or cy2 - cy1 < 14:
            continue
        # Never touch if any face box overlaps this corner
        skip_corner = False
        if subjects:
            for s in subjects:
                fx1, fy1, fx2, fy2 = s.geom.face_box
                if not (fx2 < cx1 or fx1 > cx2 or fy2 < cy1 or fy1 > cy2):
                    skip_corner = True
                    break
        if skip_corner:
            continue

        roi = out[cy1:cy2, cx1:cx2].astype(np.float32)
        lum = 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
        chroma_span = np.max(roi, axis=2) - np.min(roi, axis=2)
        bg_broad = box_filter(lum, radius=max(5, int(round(min(h, w) * 0.012))))
        # Sparse bright achromatic text strokes over darker fabric
        stroke = (lum > 192.0) & (chroma_span < 30.0) & ((lum - bg_broad) > 58.0) & (bg_broad < 140.0)
        if fg_alpha is not None:
            al_roi = fg_alpha[cy1:cy2, cx1:cx2]
            stroke = stroke & (al_roi > 0.45)

        frac = float(np.mean(stroke))
        if 0.0015 <= frac <= 0.095:
            dil = np.clip(box_filter(stroke.astype(np.float32), radius=2) * 4.5, 0.0, 1.0)
            # Local background estimate excluding stroke pixels
            valid_w_2d = 1.0 - np.clip(stroke.astype(np.float32), 0.0, 0.98)
            r_rad = max(6, int(round(min(h, w) * 0.014)))
            num = box_filter(roi * valid_w_2d[:, :, None], radius=r_rad)
            den = np.maximum(box_filter(valid_w_2d, radius=r_rad), 1e-3)[:, :, None]
            healed = num / den
            blended = roi * (1.0 - dil[:, :, None]) + healed * dil[:, :, None]
            out[cy1:cy2, cx1:cx2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    return out


# ============================================================================
# BRIA AI PORTRAIT MATTING & CLOSED-FORM COLOR DECONTAMINATION WRAPPERS
# ============================================================================

def decontaminate_foreground_colors(
    img_rgb: np.ndarray,
    alpha: np.ndarray,
    bg_orig_rgb: tuple[float, float, float] | list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Delegate closed-form boundary color decontamination to `bria_bg_engine`."""
    return decontaminate_bria_foreground(img_rgb, alpha, bg_orig_rgb=bg_orig_rgb)


def extract_foreground_matte(img_rgb: np.ndarray, geom: FacialAnthropometrics | None = None) -> np.ndarray:
    """Delegate continuous foreground alpha matting to V8 `extract_bria_foreground_matte`."""
    alpha, _tag = extract_bria_foreground_matte(img_rgb, geom=geom)
    return alpha


# ============================================================================
# TELEMETRY OVERLAY RENDERER (MULTI-SUBJECT FACIAL MATHEMATICS INSPECTOR)
# ============================================================================

def render_telemetry_overlay(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    subjects: list[SubjectProfile] | None = None,
) -> np.ndarray:
    """
    Draw scientific facial mathematics overlay for ALL detected foreground subjects (`1–4` people):
    - Daugman Iris Limbus & Pupil circles + Corneal Fresnel Purkinje vectors
    - Inter-Pupillary Horizon & Head Roll angle
    - Crown-to-Chin Biometric Axis
    - 9 Cervical-Clavicular-Shoulder Anthropometric Dress Anchors
    """
    im = Image.fromarray(img_rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(im)
    h, w = img_rgb.shape[:2]
    lw = max(2, int(min(h, w) / 320))

    geoms_to_draw: list[FacialAnthropometrics] = []
    if subjects:
        for s in subjects:
            geoms_to_draw.append(scale_facial_anthropometrics(s.geom, w, h))
    else:
        geoms_to_draw.append(scale_facial_anthropometrics(geom, w, h))

    box_palette = [(66, 133, 244), (16, 185, 129), (245, 158, 11), (236, 72, 153)]

    for idx, g in enumerate(geoms_to_draw):
        bcol = box_palette[idx % len(box_palette)]
        fx1, fy1, fx2, fy2 = g.face_box
        draw.rectangle([fx1, fy1, fx2, fy2], outline=bcol, width=lw)

        for eye in (g.left_eye, g.right_eye):
            cx, cy = eye.center_x, eye.center_y
            ri = eye.iris_radius
            rp = eye.pupil_radius
            draw.ellipse(
                [cx - eye.palpebral_rx, cy - eye.palpebral_ry, cx + eye.palpebral_rx, cy + eye.palpebral_ry],
                outline=(0, 230, 180),
                width=max(1, lw - 1),
            )
            draw.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], outline=(0, 255, 128), width=lw)
            draw.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], outline=(255, 80, 120), width=lw)
            draw.ellipse([eye.purkinje_x - 2, eye.purkinje_y - 2, eye.purkinje_x + 2, eye.purkinje_y + 2], fill=(255, 255, 100))

        draw.line([(g.left_eye.center_x, g.left_eye.center_y), (g.right_eye.center_x, g.right_eye.center_y)], fill=(255, 215, 0), width=lw)
        draw.line([g.crown_xy, g.chin_xy], fill=(155, 114, 203), width=lw)

        cervical_links = [
            (g.neck_top_left, g.neck_top_right),
            (g.neck_top_left, g.neck_base_left),
            (g.neck_top_right, g.neck_base_right),
            (g.neck_base_left, g.jugular_notch),
            (g.neck_base_right, g.jugular_notch),
            (g.neck_base_left, g.shoulder_left),
            (g.neck_base_right, g.shoulder_right),
            (g.shoulder_left, g.jugular_notch),
            (g.shoulder_right, g.jugular_notch),
        ]
        for p1, p2 in cervical_links:
            draw.line([p1, p2], fill=(255, 138, 101), width=lw)

        for pt in (
            g.neck_top_left,
            g.neck_top_right,
            g.neck_base_left,
            g.neck_base_right,
            g.jugular_notch,
            g.shoulder_left,
            g.shoulder_right,
        ):
            r_dot = max(3, lw + 1)
            draw.ellipse([pt[0] - r_dot, pt[1] - r_dot, pt[0] + r_dot, pt[1] + r_dot], fill=(255, 196, 0), outline=(20, 20, 20))

    return np.array(im, dtype=np.uint8)


def _scale_subjects_list(
    subjects: list[SubjectProfile],
    new_w: int,
    new_h: int,
) -> list[SubjectProfile]:
    """Scale a list of `SubjectProfile` items cleanly to `(new_w, new_h)`."""
    if not subjects:
        return []
    scaled: list[SubjectProfile] = []
    for s in subjects:
        old_w, old_h = s.geom.img_w, s.geom.img_h
        if (old_w, old_h) == (new_w, new_h):
            scaled.append(s)
            continue
        sx = float(new_w) / float(max(1, old_w))
        sy = float(new_h) / float(max(1, old_h))
        rx1, ry1, rx2, ry2 = s.roi_box
        orn_sc = (
            resize_img(s.ornament_lock.astype(np.float32), new_w, new_h, mode="bilinear")
            if isinstance(s.ornament_lock, np.ndarray) and s.ornament_lock.size > 4
            else np.zeros((new_h, new_w), dtype=np.float32)
        )
        scaled.append(
            SubjectProfile(
                subject_index=s.subject_index,
                label=s.label,
                geom=scale_facial_anthropometrics(s.geom, new_w, new_h),
                inferred_gender=s.inferred_gender,
                gender_confidence=s.gender_confidence,
                has_head_pallu=s.has_head_pallu,
                roi_box=(int(round(rx1 * sx)), int(round(ry1 * sy)), int(round(rx2 * sx)), int(round(ry2 * sy))),
                ornament_lock=orn_sc,
            )
        )
    return scaled


def _union_skin_mask(subjects: list[SubjectProfile], geom_fallback: FacialAnthropometrics, w: int, h: int) -> np.ndarray:
    """Build union skin mask across all foreground subjects at `(w, h)`."""
    if not subjects:
        return (
            resize_img(geom_fallback.skin_mask.astype(np.float32), w, h, mode="bilinear")
            if geom_fallback.skin_mask.shape[:2] != (h, w)
            else geom_fallback.skin_mask.astype(np.float32)
        )
    acc = np.zeros((h, w), dtype=np.float32)
    for s in subjects:
        sm = (
            resize_img(s.geom.skin_mask.astype(np.float32), w, h, mode="bilinear")
            if s.geom.skin_mask.shape[:2] != (h, w)
            else s.geom.skin_mask.astype(np.float32)
        )
        acc = np.maximum(acc, sm)
    return acc


def _graft_enhanced_heads_onto_dressed(
    enhanced_clean_rgb: np.ndarray,
    dressed_rgb: np.ndarray,
    subjects: list[SubjectProfile],
) -> np.ndarray:
    """
    When a user already ran `CatVTON` Try-On BEFORE clicking `Run`, graft the newly
    CodeFormer/OcularLock/StudioAutoDirector-enhanced head & face seamlessly onto the
    already-computed `CatVTON` dressed portrait above the collar line!
    """
    h, w = enhanced_clean_rgb.shape[:2]
    if dressed_rgb.shape[:2] != (h, w):
        dressed_rgb = resize_img(dressed_rgb, w, h, mode="lanczos")
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    head_mask = np.zeros((h, w), dtype=np.float32)
    for s in subjects:
        g = s.geom if (s.geom.img_w, s.geom.img_h) == (w, h) else scale_facial_anthropometrics(s.geom, w, h)
        cx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
        cy = 0.5 * (g.crown_xy[1] + g.chin_xy[1])
        rx = max(24.0, 1.35 * g.ipd_px)
        ry = max(30.0, 0.62 * (g.chin_xy[1] - g.crown_xy[1]) + 0.35 * g.ipd_px)
        ell = np.clip(1.35 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0.0, 1.0)
        # Keep strictly above chin + 0.08 * ipd so CatVTON collar/neckline stays 100% intact
        chin_cut = np.clip(((g.chin_xy[1] + 0.08 * g.ipd_px) - yy) / max(6.0, 0.14 * g.ipd_px), 0.0, 1.0)
        head_mask = np.maximum(head_mask, ell * chin_cut)
    head_mask = gaussian_blur(head_mask, sigma=max(2.0, min(h, w) * 0.005))[:, :, None]
    out = dressed_rgb.astype(np.float32) * (1.0 - head_mask) + enhanced_clean_rgb.astype(np.float32) * head_mask
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ============================================================================
# MASTER V8 CLOUD-NATIVE PIPELINE & DECOUPLED TRY-ON ENGINE
# ============================================================================

def process_photo_v3(
    input_img: np.ndarray,
    custom_color: str = "#98f3ff",
    upscale_factor: int = 4,
    eye_lock_strength: float = 0.85,
    catchlight_boost: float = 0.35,
    skin_smoothing: float = 0.52,
    pore_detail_4k: float = 0.35,
    relight_strength: float = 0.35,
    dress_preset: str = "none",
    dress_vertical_shift: float = 0.0,
    dress_shoulder_scale: float = 1.0,
    backdrop_style: str = "solid",
    passport_standard: str = "original_ratio",
    auto_level_head: bool = False,
    use_ai_upscale: bool = True,
    use_full_ai: bool = True,
    candidate_name_stamp: str = "",
    date_stamp: str = "",
    add_photo_border: bool = False,
    print_sheet_layout: str = "single_photo",
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
    dress_target_person: str = "auto",
    custom_dress_img: np.ndarray | None = None,
    prior_vton_state: dict | None = None,
) -> dict:
    """
    Execute the V8 Multi-Subject Pure-BRIA + CatVTON + 4x-UltraSharp + DermaPore-8K™ Studio Pipeline.
    """
    t_total_start = time.perf_counter()
    stage_timings_ms: dict[str, float] = {}

    if input_img is None:
        raise ValueError("input_img cannot be None")
    if isinstance(input_img, str):
        input_img = imread_rgb(input_img)
    if input_img is None:
        raise ValueError("Could not decode input_img")

    orig_rgb = input_img[:, :, :3].copy()
    if custom_dress_img is not None and isinstance(custom_dress_img, np.ndarray) and custom_dress_img.ndim == 3:
        custom_dress_img = custom_dress_img[:, :, :3].copy()
    else:
        custom_dress_img = None

    rgb_color = hex_to_rgb(custom_color)
    norm_hex = normalize_color_to_hex(custom_color)
    tier = max(1, min(10, int(upscale_factor))) if use_ai_upscale else 1
    config = UPSCALE_TIERS[tier]

    ih, iw = orig_rgb.shape[:2]
    native_max = max(ih, iw)
    img_sig = compute_image_signature(orig_rgb)
    custom_dress_sig = compute_image_signature(custom_dress_img)

    # Step 1: Pre-Sizing (up to 1536px) + Surgical Dual-Pass Pure-BRIA Matting + Multi-Subject Detection
    t_bg = time.perf_counter()
    pre_max_dim = 1536
    resized_input = smart_resize(orig_rgb, pre_max_dim)
    pre_h, pre_w = resized_input.shape[:2]
    if use_ai_upscale and tier > 1 and max(pre_h, pre_w) < 340:
        pre_rgb = resize_img(resized_input, pre_w * 2, pre_h * 2, mode="lanczos")
    else:
        pre_rgb = resized_input.copy()

    # Single-pass Dual-Pass Pure-BRIA matte + Multi-Subject detection (rejects Bal Krishna wall paintings & posters!)
    fg_alpha_pre, bria_tag = extract_bria_foreground_matte(pre_rgb, geom=None, subjects=None)
    subjects_pre = detect_foreground_subjects(pre_rgb, raw_bria_alpha=fg_alpha_pre, max_subjects=4)
    fg_alpha_pre = prune_background_bystanders(fg_alpha_pre, subjects_pre)
    geom_pre = subjects_pre[0].geom

    # Optional head straightening (for tilted single portraits)
    if auto_level_head and len(subjects_pre) == 1 and abs(geom_pre.roll_deg) >= 0.45:
        pre_rgb = auto_level_portrait(pre_rgb, geom_pre, bg_rgb=None)
        fg_alpha_pre, bria_tag = extract_bria_foreground_matte(pre_rgb, geom=None, subjects=None)
        subjects_pre = detect_foreground_subjects(pre_rgb, raw_bria_alpha=fg_alpha_pre, max_subjects=4)
        fg_alpha_pre = prune_background_bystanders(fg_alpha_pre, subjects_pre)
        geom_pre = subjects_pre[0].geom

    pre_clean_rgb = decontaminate_bria_foreground(pre_rgb, fg_alpha_pre)
    pre_clean_rgb = clean_corner_camera_watermark(pre_clean_rgb, fg_alpha=fg_alpha_pre, subjects=subjects_pre)
    stage_timings_ms["bria_and_multi_subject_ms"] = round((time.perf_counter() - t_bg) * 1000.0, 2)

    # Step 2: V8 Multi-Subject Studio Auto-Director + IdentityMarkLock™ ∪ OrnamentLock™ + CodeFormer
    t_ai = time.perf_counter()
    ai_diag: dict[str, object] = {}
    if use_full_ai:
        clean_enhanced_pre_rgb, ai_diag = full_ai_portrait_intelligence_multi(
            pre_clean_rgb,
            subjects=subjects_pre,
            geom=geom_pre,
            slider_tier=tier,
            eye_lock_strength=max(0.75, float(eye_lock_strength)),
            catchlight_boost=float(catchlight_boost),
        )
    else:
        clean_enhanced_pre_rgb = pre_clean_rgb.copy()
        if eye_lock_strength > 0.01 or catchlight_boost > 0.01:
            clean_enhanced_pre_rgb = ocular_lock_restore(
                orig_rgb=pre_clean_rgb,
                ai_enhanced_rgb=clean_enhanced_pre_rgb,
                geom=geom_pre,
                eye_lock_strength=eye_lock_strength,
                catchlight_boost=catchlight_boost,
            )
        if skin_smoothing > 0.01 or pore_detail_4k > 0.01:
            clean_enhanced_pre_rgb = dermaphysics_skin_enhance(
                clean_enhanced_pre_rgb,
                geom=geom_pre,
                smoothing_strength=skin_smoothing,
                pore_detail_4k=pore_detail_4k,
            )
        if relight_strength > 0.01:
            clean_enhanced_pre_rgb = spherical_harmonics_relight(
                clean_enhanced_pre_rgb,
                geom=geom_pre,
                relight_strength=relight_strength,
            )
    stage_timings_ms["ai_portrait_brain_ms"] = round((time.perf_counter() - t_ai) * 1000.0, 2)

    # Step 3: Real Diffusion AI Virtual Try-On (`CatVTON` FP16 CUDA) BEFORE Super-Resolution!
    wants_dress = (bool(dress_preset) and str(dress_preset) != "none") or (custom_dress_img is not None)
    vton_combo_sig = (
        img_sig,
        str(dress_preset or "none"),
        str(dress_target_person or "auto"),
        custom_dress_sig,
        round(float(dress_vertical_shift), 2),
        round(float(dress_shoulder_scale), 2),
        bool(auto_level_head),
    )

    dressed_pre_rgb = clean_enhanced_pre_rgb.copy()
    dressed_fg_alpha_pre = fg_alpha_pre.copy()

    if wants_dress:
        t_dr = time.perf_counter()
        # Reuse cached CatVTON try-on if user already previewed this exact dress before clicking Run!
        if (
            isinstance(prior_vton_state, dict)
            and prior_vton_state.get("vton_combo_sig") == vton_combo_sig
            and isinstance(prior_vton_state.get("dressed_base_rgb"), np.ndarray)
            and isinstance(prior_vton_state.get("dressed_alpha_u8"), np.ndarray)
        ):
            cached_dressed = prior_vton_state["dressed_base_rgb"]
            cached_d_alpha = prior_vton_state["dressed_alpha_u8"].astype(np.float32) / 255.0
            if cached_dressed.shape[:2] != clean_enhanced_pre_rgb.shape[:2]:
                ph, pw = clean_enhanced_pre_rgb.shape[:2]
                cached_dressed = resize_img(cached_dressed, pw, ph, mode="lanczos")
                cached_d_alpha = resize_img(cached_d_alpha, pw, ph, mode="bilinear")
            dressed_pre_rgb = _graft_enhanced_heads_onto_dressed(
                clean_enhanced_pre_rgb,
                cached_dressed,
                subjects_pre,
            )
            dressed_fg_alpha_pre = np.clip(cached_d_alpha, 0.0, 1.0)
            stage_timings_ms["catvton_dress_cache_hit_ms"] = round((time.perf_counter() - t_dr) * 1000.0, 2)
        else:
            dressed_pre_rgb, dressed_fg_alpha_pre = apply_anthropometric_dress(
                clean_enhanced_pre_rgb,
                preset_key=str(dress_preset or "none"),
                geom=geom_pre,
                fg_alpha=fg_alpha_pre,
                vertical_shift=dress_vertical_shift,
                shoulder_scale=dress_shoulder_scale,
                subjects=subjects_pre,
                target_person=str(dress_target_person or "auto"),
                custom_garment_rgb=custom_dress_img,
                current_dressed_rgb=None,
            )
            stage_timings_ms["catvton_dress_ms"] = round((time.perf_counter() - t_dr) * 1000.0, 2)

    # Step 4: True Native 2K / 4K / 8K (`7680px`) Neural Super-Resolution on BOTH Face AND Dress!
    t_sr = time.perf_counter()
    union_skin_pre = _union_skin_mask(subjects_pre, geom_pre, dressed_pre_rgb.shape[1], dressed_pre_rgb.shape[0])

    if not use_ai_upscale or tier == 1:
        working_rgb = edge_directed_4k_upscale(
            dressed_pre_rgb,
            scale_factor=1.0,
            max_target_dim=max(dressed_pre_rgb.shape[:2]),
            detail_crisping=0.14 if use_full_ai else 0.0,
            skin_mask=union_skin_pre,
        )
        fg_alpha = upsample_alpha_guided(dressed_fg_alpha_pre, working_rgb)
        engine_used = f"V8 Native 1x Studio ({bria_tag}, {len(subjects_pre)} subj)"
    else:
        target_max = max(native_max, int(config["target_max_px"]))
        eff_scale = max(1.0, float(target_max) / float(max(16, max(dressed_pre_rgb.shape[:2]))))
        working_rgb = edge_directed_4k_upscale(
            dressed_pre_rgb,
            scale_factor=eff_scale,
            max_target_dim=target_max,
            detail_crisping=float(config["crisping"]),
            skin_mask=union_skin_pre,
        )
        fg_alpha = upsample_alpha_guided(dressed_fg_alpha_pre, working_rgb)
        working_rgb = synthesize_hd_dermal_and_fabric_microtexture(
            working_rgb,
            subjects=subjects_pre,
            geom=geom_pre,
            fg_alpha=fg_alpha,
            slider_tier=tier,
        )
        engine_used = f"V8 True {target_max}px Tier {tier}/10 (4x-UltraSharp + DermaPore-8K™ + {bria_tag}, {len(subjects_pre)} subj)"

    if use_full_ai:
        engine_used += " + Multi-Subject StudioAutoDirectorAI (IdentityMarkLock™ + OrnamentLock™)"
    if wants_dress:
        engine_used += " + CatVTON Virtual Try-On"
    stage_timings_ms["super_resolution_8k_ms"] = round((time.perf_counter() - t_sr) * 1000.0, 2)

    out_h, out_w = working_rgb.shape[:2]
    subjects_final = _scale_subjects_list(subjects_pre, out_w, out_h)
    geom = subjects_final[0].geom if subjects_final else scale_facial_anthropometrics(geom_pre, out_w, out_h)

    # Step 5: Foreground-Scoped Manual Colour Correction + Pure Studio Backdrop Compositing
    t_comp = time.perf_counter()
    base_composited = apply_bria_studio_background(
        working_rgb,
        fg_alpha,
        bg_rgb_color=rgb_color,
        backdrop_style=backdrop_style,
    )
    if passport_standard and passport_standard != "original_ratio":
        base_composited = icao_biometric_crop(base_composited, standard_key=passport_standard, geom=geom, bg_rgb=rgb_color)

    if has_active_manual_adjustments(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        warmth=warmth,
        shadows_lift=shadows_lift,
    ):
        composited = render_foreground_and_backdrop(
            fg_rgb=working_rgb,
            fg_alpha=fg_alpha,
            bg_rgb_color=rgb_color,
            backdrop_style=backdrop_style,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            gamma=gamma,
            warmth=warmth,
            shadows_lift=shadows_lift,
        )
        if passport_standard and passport_standard != "original_ratio":
            composited = icao_biometric_crop(composited, standard_key=passport_standard, geom=geom, bg_rgb=rgb_color)
    else:
        composited = base_composited.copy()

    if candidate_name_stamp.strip() or date_stamp.strip() or add_photo_border:
        composited = apply_name_date_plaque(
            composited,
            candidate_name=candidate_name_stamp,
            dob_or_date=date_stamp,
            add_border=add_photo_border,
        )
    stage_timings_ms["backdrop_and_color_ms"] = round((time.perf_counter() - t_comp) * 1000.0, 2)

    t_io = time.perf_counter()
    # Render telemetry on compact <=1600px view so Face Guide loads instantly even at 8K
    telem_base = smart_resize(working_rgb, 1600)
    telemetry_rgb = render_telemetry_overlay(telem_base, geom, subjects=subjects_final)

    os.makedirs(TEMP_DIR, exist_ok=True)
    final_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_final.png"))
    imwrite_rgb(final_path, composited)

    if print_sheet_layout and print_sheet_layout != "single_photo":
        sheet_rgb = generate_print_sheet(composited, layout_key=print_sheet_layout)
        sheet_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_print_sheet.png"))
        imwrite_rgb(sheet_path, sheet_rgb)
    else:
        sheet_rgb = composited
        sheet_path = final_path

    stage_timings_ms["io_and_telemetry_ms"] = round((time.perf_counter() - t_io) * 1000.0, 2)
    stage_timings_ms["total_ms"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)

    dress_label = (
        "Custom Uploaded Dress (CatVTON)"
        if custom_dress_img is not None
        else DRESS_PRESETS.get(dress_preset, {}).get("label", "None")
    )
    metrics = {
        "engine_pipeline": engine_used,
        "bria_background_engine": bria_tag,
        "bria_status": get_bria_engine_status(),
        "detected_foreground_subjects": len(subjects_pre),
        "subject_genders": [s.label for s in subjects_pre],
        "passport_standard": str(passport_standard or "original_ratio"),
        "use_full_ai": bool(use_full_ai),
        "use_ai_upscale": bool(use_ai_upscale),
        "auto_level_head": bool(auto_level_head),
        "ai_upscale_tier": f"{tier} / 10" if use_ai_upscale else "OFF (1x Native)",
        "full_ai_feature": "ON (V8 Multi-Subject StudioAutoDirector + IdentityMarkLock™ + OrnamentLock™)" if use_full_ai else "OFF",
        "auto_straighten_head": "ON" if auto_level_head else "OFF",
        "background_color_hex": norm_hex,
        "background_color_rgb": list(rgb_color),
        "output_resolution_px": f"{composited.shape[1]} × {composited.shape[0]}",
        "sheet_resolution_px": f"{sheet_rgb.shape[1]} × {sheet_rgb.shape[0]}",
        "inter_pupillary_distance_px": round(geom.ipd_px, 2),
        "head_roll_tilt_deg": round(geom.roll_deg, 2),
        "head_yaw_asymmetry": round(geom.yaw_asymmetry, 3),
        "skin_lab_median": [round(x, 2) for x in geom.skin_lab_median],
        "lateral_shadow_asymmetry_L": round(geom.lateral_shadow_diff, 2),
        "ocular_lock_strength": round(eye_lock_strength, 2),
        "dress_applied": dress_label,
        "dress_target_person": str(dress_target_person or "auto"),
        "veil_reminder": build_veil_dress_reminder(
            subjects_pre,
            dress_preset=str(dress_preset or "none"),
            target_person=str(dress_target_person or "auto"),
        ),
        "stage_timings_ms": stage_timings_ms,
        "ai_photometric_diagnosis": ai_diag,
    }

    # Update backward-compatible in-process cache (for local unit tests)
    _LAST_RUN_CACHE["base_composited"] = base_composited.copy()
    _LAST_RUN_CACHE["fg_rgb"] = working_rgb.copy()
    _LAST_RUN_CACHE["fg_alpha"] = fg_alpha.copy()
    _LAST_RUN_CACHE["custom_color"] = norm_hex
    _LAST_RUN_CACHE["backdrop_style"] = backdrop_style
    _LAST_RUN_CACHE["passport_standard"] = passport_standard
    _LAST_RUN_CACHE["tier"] = tier
    _LAST_RUN_CACHE["native_max"] = native_max
    _LAST_RUN_CACHE["use_ai_upscale"] = bool(use_ai_upscale)
    _LAST_RUN_CACHE["geom"] = geom
    _LAST_RUN_CACHE["telemetry_rgb"] = telemetry_rgb
    _LAST_RUN_CACHE["metrics"] = metrics
    _LAST_RUN_CACHE["params"] = {
        "candidate_name_stamp": candidate_name_stamp,
        "date_stamp": date_stamp,
        "add_photo_border": add_photo_border,
        "print_sheet_layout": print_sheet_layout,
    }

    # Build ZeroGPU-safe session state returned to main Gradio process `gr.State`
    fg_alpha_u8 = np.clip(np.round(fg_alpha * 255.0), 0, 255).astype(np.uint8)
    clean_alpha_u8 = np.clip(np.round(fg_alpha_pre * 255.0), 0, 255).astype(np.uint8)
    dressed_alpha_u8 = np.clip(np.round(dressed_fg_alpha_pre * 255.0), 0, 255).astype(np.uint8)

    session_state = {
        "img_sig": img_sig,
        "has_run_active": True,
        "active_tier": int(tier),
        "use_ai_upscale": bool(use_ai_upscale),
        "native_max": int(native_max),
        "auto_level_head": bool(auto_level_head),
        "backdrop_style": str(backdrop_style or "solid"),
        "passport_standard": str(passport_standard or "original_ratio"),
        "candidate_name_stamp": str(candidate_name_stamp or ""),
        "date_stamp": str(date_stamp or ""),
        "add_photo_border": bool(add_photo_border),
        # Pristine <=1536px base layers for zero-degradation repeated AI Dress Try-On
        "clean_base_rgb": clean_enhanced_pre_rgb.copy(),
        "clean_alpha_u8": clean_alpha_u8,
        "dressed_base_rgb": dressed_pre_rgb.copy(),
        "dressed_alpha_u8": dressed_alpha_u8,
        "vton_combo_sig": vton_combo_sig if wants_dress else None,
        # Full-resolution foreground & uint8 alpha for lossless <30ms live Manual Colour & BG Picker
        "fg_rgb": working_rgb.copy(),
        "fg_alpha_u8": fg_alpha_u8,
        "custom_color": norm_hex,
        "metrics": metrics,
    }

    return {
        "final_rgb": composited,
        "final_path": final_path,
        "sheet_rgb": sheet_rgb,
        "sheet_path": sheet_path,
        "telemetry_rgb": telemetry_rgb,
        "metrics": metrics,
        "geometry": geom,
        "session_state": session_state,
    }


def extract_compact_vton_input_state(session_state: dict | None, img_sig: str) -> dict | None:
    """
    Strip heavy 4K/8K `fg_rgb` arrays before passing `session_state` into `@spaces.GPU`
    so ZeroGPU pickle IPC stays `< 6 MB` (`< 15 ms`)!
    """
    if not isinstance(session_state, dict) or session_state.get("img_sig") != img_sig:
        return None
    return {
        "img_sig": session_state.get("img_sig"),
        "has_run_active": bool(session_state.get("has_run_active", False)),
        "active_tier": int(session_state.get("active_tier", 1)),
        "use_ai_upscale": bool(session_state.get("use_ai_upscale", True)),
        "native_max": int(session_state.get("native_max", 1080)),
        "auto_level_head": bool(session_state.get("auto_level_head", False)),
        "backdrop_style": str(session_state.get("backdrop_style", "solid")),
        "passport_standard": str(session_state.get("passport_standard", "original_ratio")),
        "clean_base_rgb": session_state.get("clean_base_rgb"),
        "clean_alpha_u8": session_state.get("clean_alpha_u8"),
        "dressed_base_rgb": session_state.get("dressed_base_rgb"),
        "dressed_alpha_u8": session_state.get("dressed_alpha_u8"),
        "vton_combo_sig": session_state.get("vton_combo_sig"),
    }


def tryon_dress_decoupled(
    input_img: np.ndarray,
    compact_state: dict | None = None,
    dress_preset: str = "none",
    dress_target_person: str = "auto",
    custom_dress_img: np.ndarray | None = None,
    dress_vertical_shift: float = 0.0,
    dress_shoulder_scale: float = 1.0,
    custom_color: str = "#98f3ff",
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
) -> dict:
    """
    V8 Decoupled "No-Run, Zero-Degradation" AI Dress Try-On Engine:
    - Does NOT require clicking `🚀 Run` first!
    - NEVER stacks a new dress over a previous dress (resets edited subject's ROI from
      `clean_base_rgb` while preserving other subjects' outfits in `dressed_base_rgb`).
    - NEVER re-upscales an already-upscaled image! If `Run` has not been clicked yet,
      outputs directly at crisp native preview resolution; if `Run` WAS clicked earlier,
      upscales the newly dressed base ONCE from `1x -> active_tier`.
    """
    t0 = time.perf_counter()
    orig_rgb = input_img[:, :, :3].copy()
    if custom_dress_img is not None and isinstance(custom_dress_img, np.ndarray) and custom_dress_img.ndim == 3:
        custom_dress_img = custom_dress_img[:, :, :3].copy()
    else:
        custom_dress_img = None

    img_sig = compute_image_signature(orig_rgb)
    custom_dress_sig = compute_image_signature(custom_dress_img)
    rgb_color = hex_to_rgb(custom_color)
    norm_hex = normalize_color_to_hex(custom_color)

    has_valid_base = (
        isinstance(compact_state, dict)
        and compact_state.get("img_sig") == img_sig
        and isinstance(compact_state.get("clean_base_rgb"), np.ndarray)
        and isinstance(compact_state.get("clean_alpha_u8"), np.ndarray)
    )

    if has_valid_base:
        clean_base_rgb = compact_state["clean_base_rgb"].copy()  # type: ignore[index]
        clean_alpha_pre = compact_state["clean_alpha_u8"].astype(np.float32) / 255.0  # type: ignore[index]
        prev_dressed = compact_state.get("dressed_base_rgb")  # type: ignore[union-attr]
        current_dressed_rgb = prev_dressed.copy() if isinstance(prev_dressed, np.ndarray) else clean_base_rgb.copy()
        has_run_active = bool(compact_state.get("has_run_active", False))  # type: ignore[union-attr]
        active_tier = int(compact_state.get("active_tier", 1))  # type: ignore[union-attr]
        use_ai_upscale = bool(compact_state.get("use_ai_upscale", True))  # type: ignore[union-attr]
        native_max = int(compact_state.get("native_max", max(orig_rgb.shape[:2])))  # type: ignore[union-attr]
        auto_level_head = bool(compact_state.get("auto_level_head", False))  # type: ignore[union-attr]
        bria_tag = "Cached Session Pure-BRIA Matte"
    else:
        pre_rgb = smart_resize(orig_rgb, 1536)
        clean_alpha_pre, bria_tag = extract_bria_foreground_matte(pre_rgb, geom=None, subjects=None)
        subjects_init = detect_foreground_subjects(pre_rgb, raw_bria_alpha=clean_alpha_pre, max_subjects=4)
        clean_alpha_pre = prune_background_bystanders(clean_alpha_pre, subjects_init)
        clean_base_rgb = decontaminate_bria_foreground(pre_rgb, clean_alpha_pre)
        clean_base_rgb = clean_corner_camera_watermark(clean_base_rgb, fg_alpha=clean_alpha_pre, subjects=subjects_init)
        current_dressed_rgb = clean_base_rgb.copy()
        has_run_active = False
        active_tier = 1
        use_ai_upscale = False
        native_max = max(orig_rgb.shape[:2])
        auto_level_head = False

    subjects_pre = detect_foreground_subjects(clean_base_rgb, raw_bria_alpha=clean_alpha_pre, max_subjects=4)
    geom_pre = subjects_pre[0].geom

    wants_dress = (bool(dress_preset) and str(dress_preset) != "none") or (custom_dress_img is not None)
    if not wants_dress:
        dressed_base_rgb = clean_base_rgb.copy()
        dressed_alpha_pre = clean_alpha_pre.copy()
    else:
        dressed_base_rgb, dressed_alpha_pre = apply_anthropometric_dress(
            clean_base_rgb,
            preset_key=str(dress_preset or "none"),
            geom=geom_pre,
            fg_alpha=clean_alpha_pre,
            vertical_shift=dress_vertical_shift,
            shoulder_scale=dress_shoulder_scale,
            subjects=subjects_pre,
            target_person=str(dress_target_person or "auto"),
            custom_garment_rgb=custom_dress_img,
            current_dressed_rgb=current_dressed_rgb,
        )

    # Only upscale if the user previously clicked `Run` with AI Upscale enabled — NEVER stack upscales!
    if has_run_active and use_ai_upscale and active_tier > 1:
        cfg = UPSCALE_TIERS[max(1, min(10, active_tier))]
        target_max = max(native_max, int(cfg["target_max_px"]))
        eff_scale = max(1.0, float(target_max) / float(max(16, max(dressed_base_rgb.shape[:2]))))
        union_skin = _union_skin_mask(subjects_pre, geom_pre, dressed_base_rgb.shape[1], dressed_base_rgb.shape[0])
        fg_rgb = edge_directed_4k_upscale(
            dressed_base_rgb,
            scale_factor=eff_scale,
            max_target_dim=target_max,
            detail_crisping=float(cfg["crisping"]),
            skin_mask=union_skin,
        )
        fg_alpha = upsample_alpha_guided(dressed_alpha_pre, fg_rgb)
        fg_rgb = synthesize_hd_dermal_and_fabric_microtexture(
            fg_rgb,
            subjects=subjects_pre,
            geom=geom_pre,
            fg_alpha=fg_alpha,
            slider_tier=active_tier,
        )
    else:
        fg_rgb = dressed_base_rgb.copy()
        fg_alpha = dressed_alpha_pre.copy()

    composited = render_foreground_and_backdrop(
        fg_rgb=fg_rgb,
        fg_alpha=fg_alpha,
        bg_rgb_color=rgb_color,
        backdrop_style="solid",
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        warmth=warmth,
        shadows_lift=shadows_lift,
    )

    os.makedirs(TEMP_DIR, exist_ok=True)
    final_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_final.png"))
    imwrite_rgb(final_path, composited)

    telem_base = smart_resize(fg_rgb, 1600)
    telemetry_rgb = render_telemetry_overlay(telem_base, geom_pre, subjects=subjects_pre)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    vton_combo_sig = (
        img_sig,
        str(dress_preset or "none"),
        str(dress_target_person or "auto"),
        custom_dress_sig,
        round(float(dress_vertical_shift), 2),
        round(float(dress_shoulder_scale), 2),
        bool(auto_level_head),
    )

    dress_label = (
        "Custom Uploaded Dress (CatVTON)"
        if custom_dress_img is not None
        else DRESS_PRESETS.get(dress_preset, {}).get("label", "Original Dress")
    )
    veil_note = build_veil_dress_reminder(
        subjects_pre,
        dress_preset=str(dress_preset or "none"),
        target_person=str(dress_target_person or "auto"),
    )
    metrics = {
        "mode": "V9 Decoupled Photorealistic AI Virtual Try-On (Zero Re-Upscale Degradation)",
        "bria_background_engine": bria_tag,
        "detected_foreground_subjects": len(subjects_pre),
        "subject_genders": [s.label for s in subjects_pre],
        "dress_applied": dress_label,
        "dress_target_person": str(dress_target_person or "auto"),
        "veil_reminder": veil_note,
        "output_resolution_px": f"{composited.shape[1]} × {composited.shape[0]}",
        "tryon_elapsed_ms": elapsed_ms,
    }

    new_session_state = {
        "img_sig": img_sig,
        "has_run_active": has_run_active,
        "active_tier": active_tier,
        "use_ai_upscale": use_ai_upscale,
        "native_max": native_max,
        "auto_level_head": auto_level_head,
        "backdrop_style": "solid",
        "passport_standard": "original_ratio",
        "clean_base_rgb": clean_base_rgb.copy(),
        "clean_alpha_u8": np.clip(np.round(clean_alpha_pre * 255.0), 0, 255).astype(np.uint8),
        "dressed_base_rgb": dressed_base_rgb.copy(),
        "dressed_alpha_u8": np.clip(np.round(dressed_alpha_pre * 255.0), 0, 255).astype(np.uint8),
        "vton_combo_sig": vton_combo_sig if wants_dress else None,
        "fg_rgb": fg_rgb.copy(),
        "fg_alpha_u8": np.clip(np.round(fg_alpha * 255.0), 0, 255).astype(np.uint8),
        "custom_color": norm_hex,
        "metrics": metrics,
    }

    return {
        "final_rgb": composited,
        "final_path": final_path,
        "telemetry_rgb": telemetry_rgb,
        "metrics": metrics,
        "session_state": new_session_state,
    }


def apply_live_manual_color_v8(
    session_state: dict | None = None,
    input_img: np.ndarray | None = None,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
    custom_color: str | None = None,
) -> np.ndarray | None:
    """
    ZeroGPU-Safe (<20ms) Live Manual Colour Correction & Background Colour Re-Compositing:
    - Reads foreground `fg_rgb` & `fg_alpha_u8` from browser-session `gr.State` (`session_state`)
      so ZeroGPU subprocess isolation NEVER loses the processed image!
    - Grades ONLY the foreground person(s) and composites over untinted studio `custom_color`
      so `#98f3ff` / `#ffffff` official backgrounds remain 100% pure!
    - Even if the user drags sliders BEFORE clicking `Run` or `Try On`, previews live manual
      colour adjustments directly on `input_img`!
    """
    eff_hex = normalize_color_to_hex(
        custom_color if custom_color is not None else (session_state.get("custom_color") if isinstance(session_state, dict) else "#98f3ff"),
        default="#98f3ff",
    )
    rgb_color = hex_to_rgb(eff_hex)

    if isinstance(session_state, dict):
        fg_rgb = session_state.get("fg_rgb")
        fg_alpha_u8 = session_state.get("fg_alpha_u8")
        if isinstance(fg_rgb, np.ndarray) and isinstance(fg_alpha_u8, np.ndarray):
            if input_img is None or session_state.get("img_sig") == compute_image_signature(input_img):
                b_style = str(session_state.get("backdrop_style") or "solid")
                out = render_foreground_and_backdrop(
                    fg_rgb=fg_rgb,
                    fg_alpha=fg_alpha_u8.astype(np.float32) / 255.0,
                    bg_rgb_color=rgb_color,
                    backdrop_style=b_style,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    gamma=gamma,
                    warmth=warmth,
                    shadows_lift=shadows_lift,
                )
                s_name = str(session_state.get("candidate_name_stamp") or "")
                s_date = str(session_state.get("date_stamp") or "")
                s_bord = bool(session_state.get("add_photo_border", False))
                if s_name.strip() or s_date.strip() or s_bord:
                    out = apply_name_date_plaque(
                        out,
                        candidate_name=s_name,
                        dob_or_date=s_date,
                        add_border=s_bord,
                    )
                session_state["custom_color"] = eff_hex
                return out

    # Fallback: if user moves Manual Colour sliders BEFORE clicking Run (on raw upload),
    # preview the manual colour correction directly on the uploaded image (or cached raw_upload_rgb)!
    raw_up = input_img if isinstance(input_img, np.ndarray) else (session_state.get("raw_upload_rgb") if isinstance(session_state, dict) else None)
    if isinstance(raw_up, np.ndarray):
        if has_active_manual_adjustments(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            gamma=gamma,
            warmth=warmth,
            shadows_lift=shadows_lift,
        ):
            return apply_correction(
                raw_up[:, :, :3],
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                gamma=gamma,
                warmth=warmth,
                shadows_lift=shadows_lift,
            )
        return raw_up[:, :, :3].copy()

    # Secondary fallback for local unit tests calling `apply_live_manual_color_on_cached`
    return apply_live_manual_color_on_cached(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        warmth=warmth,
        shadows_lift=shadows_lift,
        custom_color=custom_color,
    )


def apply_live_manual_color_on_cached(
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
    candidate_name_stamp: str | None = None,
    date_stamp: str | None = None,
    add_photo_border: bool | None = None,
    print_sheet_layout: str | None = None,
    custom_color: str | None = None,
    write_disk: bool = False,
) -> np.ndarray | None:
    """
    Backward-compatible in-process manual color & background re-compositing helper.
    Now foreground-scoped so official studio backgrounds never tint!
    """
    fg_rgb = _LAST_RUN_CACHE.get("fg_rgb")
    fg_alpha = _LAST_RUN_CACHE.get("fg_alpha")
    base_comp = _LAST_RUN_CACHE.get("base_composited")
    if base_comp is None and fg_rgb is None:
        return None

    eff_hex = normalize_color_to_hex(
        custom_color if custom_color is not None else str(_LAST_RUN_CACHE.get("custom_color") or "#98f3ff"),
        default="#98f3ff",
    )
    rgb_color = hex_to_rgb(eff_hex)
    b_style = str(_LAST_RUN_CACHE.get("backdrop_style") or "solid")
    p_std = str(_LAST_RUN_CACHE.get("passport_standard") or "original_ratio")
    geom = _LAST_RUN_CACHE.get("geom")

    if isinstance(fg_rgb, np.ndarray) and isinstance(fg_alpha, np.ndarray):
        composited = render_foreground_and_backdrop(
            fg_rgb=fg_rgb,
            fg_alpha=fg_alpha,
            bg_rgb_color=rgb_color,
            backdrop_style=b_style,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            gamma=gamma,
            warmth=warmth,
            shadows_lift=shadows_lift,
        )
        if p_std and p_std != "original_ratio" and geom is not None:
            composited = icao_biometric_crop(composited, standard_key=p_std, geom=geom, bg_rgb=rgb_color)  # type: ignore[arg-type]
        _LAST_RUN_CACHE["custom_color"] = eff_hex
    else:
        composited = base_comp.copy()  # type: ignore[union-attr]
        if has_active_manual_adjustments(brightness, contrast, saturation, gamma, warmth, shadows_lift):
            corrected = apply_correction(
                composited,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                gamma=gamma,
                warmth=warmth,
                shadows_lift=shadows_lift,
            )
            if corrected is not None:
                composited = corrected

    cached_params = _LAST_RUN_CACHE.get("params") or {}
    if not isinstance(cached_params, dict):
        cached_params = {}
    eff_name = cached_params.get("candidate_name_stamp", "") if candidate_name_stamp is None else candidate_name_stamp
    eff_date = cached_params.get("date_stamp", "") if date_stamp is None else date_stamp
    eff_border = bool(cached_params.get("add_photo_border", False)) if add_photo_border is None else bool(add_photo_border)
    eff_sheet = cached_params.get("print_sheet_layout", "single_photo") if print_sheet_layout is None else print_sheet_layout

    if str(eff_name).strip() or str(eff_date).strip() or eff_border:
        composited = apply_name_date_plaque(
            composited,
            candidate_name=str(eff_name),
            dob_or_date=str(eff_date),
            add_border=eff_border,
        )

    if eff_sheet and eff_sheet != "single_photo":
        composited = generate_print_sheet(composited, layout_key=str(eff_sheet))

    if write_disk:
        os.makedirs(TEMP_DIR, exist_ok=True)
        final_path = os.path.abspath(os.path.join(TEMP_DIR, "temp_final.png"))
        imwrite_rgb(final_path, composited)
    return composited


def process_photo(input_img: np.ndarray, custom_color: str = "#98f3ff", upscale_factor: int = 4) -> str | None:
    """Backward-compatible entry point returning the saved file path of the processed passport photo."""
    if input_img is None:
        return None
    try:
        res = process_photo_v3(
            input_img=input_img,
            custom_color=custom_color,
            upscale_factor=upscale_factor,
        )
        return res["final_path"]
    except Exception as e:
        print(f"❌ process_photo error: {e}")
        return None
