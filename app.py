"""
Maa Sitala Passport Photo Studio V8 — Multi-Subject, Pure-BRIA, CatVTON & True 8K Cloud Master.

V8 Highlights:
1. Multi-Subject & Official Couple Portrait Intelligence (`1–4` Foreground People):
   - Understands both single portraits AND official joint/couple photos (e.g., Husband & Wife),
     rejecting wall paintings (Bal Krishna wall frames) and pruning background passersby.
   - Restores & relights every foreground person independently with `IdentityMarkLock™ ∪ OrnamentLock™`
     (protecting bindis, sindoor, nose-pins, earrings, moles, and scars).
2. Surgical Dual-Pass Pure-BRIA (`RMBG-2.0`) Background Removal (zero oval cheek/jaw halos).
3. Real Diffusion AI Virtual Try-On (`CatVTON` FP16 CUDA) + Decoupled "No-Run, Zero-Degradation" UX:
   - Selecting a dress or clicking `👗 Try On Dress` runs ONLY AI Virtual Try-On without requiring
     `🚀 Run` and NEVER stacks upscales or degrades image sharpness!
   - Supports Single & Official Couple Dress Presets, `Auto / Person 1 / Person 2 / Both` targeting,
     and Custom Uploaded Dress Photos.
4. True Native 4K (`3840px`) & 8K (`7680px`) Super-Resolution (`4x-UltraSharp` + `DermaPore-8K™`
   cellular skin pores + `WeaveSharp-8K™` garment weave clarity) on both skin AND clothing.
5. ZeroGPU-Safe Browser Session State (`gr.State`) + Foreground-Scoped Manual Colour Correction
   (`Gamma`, `Brightness`, `Contrast`, `Saturation`, `Warmth`, `Shadows Lift`, `↺ Reset`, `Colour Picker`)
   updating in `<20ms` without tinting official `#98f3ff` / `#ffffff` studio backgrounds, plus clean
   lossless PNG download button on `✨ Live Preview` (`interactive=False`).
"""
from __future__ import annotations

import inspect
import os
import threading
import numpy as np

# Guard against gradio_client + Pydantic 2.11+ boolean JSON schema bug (`TypeError: argument of type 'bool' is not iterable`)
# Must run BEFORE importing `spaces` or `gradio` so all internal bindings pick up the guarded functions.
try:
    import gradio_client.utils as _gc_utils  # type: ignore

    _orig_get_type = getattr(_gc_utils, "get_type", None)
    if callable(_orig_get_type):
        def _safe_get_type(schema):
            if isinstance(schema, bool) or not isinstance(schema, dict):
                return "Any"
            try:
                return _orig_get_type(schema)
            except Exception:
                return "Any"
        _gc_utils.get_type = _safe_get_type

    _orig_json_schema = getattr(_gc_utils, "_json_schema_to_python_type", None)
    if callable(_orig_json_schema):
        def _safe_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool) or not isinstance(schema, dict):
                return "Any"
            try:
                return _orig_json_schema(schema, defs)
            except Exception:
                return "Any"
        _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

    _orig_public_json_schema = getattr(_gc_utils, "json_schema_to_python_type", None)
    if callable(_orig_public_json_schema):
        def _safe_public_json_schema(schema):
            if isinstance(schema, bool) or not isinstance(schema, dict):
                return "Any"
            try:
                return _orig_public_json_schema(schema)
            except Exception:
                return "Any"
        _gc_utils.json_schema_to_python_type = _safe_public_json_schema
except Exception:
    pass

# Safe ZeroGPU Decorator Import (official spaces package or transparent pass-through)
try:
    import spaces  # type: ignore
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(func=None, **_kwargs):
            if func is not None and callable(func):
                return func
            def _decorator(f):
                return f
            return _decorator
    spaces = _SpacesFallback()

try:
    import gradio as gr  # type: ignore
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False

from src.shared.config import PASSPORT_STANDARDS, IS_HF_CLOUD
from src.core.codeformer_setup import setup_codeformer
from src.core.neural_hub import warmup_neural_models
from src.core.bria_bg_engine import warmup_bria_cloud_models
from src.core.vton_ai_engine import warmup_vton_cloud_models
from src.core.cv_ops import imread_rgb
from src.core.multi_subject_detector import build_veil_dress_reminder, detect_foreground_subjects
from src.core.ai_processor import (
    _LAST_RUN_CACHE,
    apply_live_manual_color_on_cached,
    apply_live_manual_color_v8,
    compute_image_signature,
    extract_compact_vton_input_state,
    normalize_color_to_hex,
    process_photo_v3,
    smart_resize,
    tryon_dress_decoupled,
    write_fast_unique_png,
)
from src.core.dress_engine import DRESS_PRESETS
from src.shared.cloud_database import (
    delete_customer,
    get_all_gallery,
    load_permanent_customer_recipe,
    pull_from_cloud,
    save_customer_recipe,
    search_customers_detailed,
    search_database,
)
from src.shared.backup_manager import (
    create_backup_file_on_disk,
    get_system_status,
    purge_system,
)
from src.shared.mobile_qr_and_share import (
    create_customer_share,
    create_mobile_shoot_session,
    get_customer_share_meta,
    poll_mobile_shoot_photo,
    poll_mobile_shoot_photo_status,
    save_mobile_shoot_photo,
    verify_customer_share_pin,
)


@spaces.GPU(duration=20)
def _zerogpu_warmup_probe() -> str:
    """Registered ZeroGPU probe so ZeroGPU hardware containers validate cleanly at startup."""
    return "ready"


# ============================================================================
# 1. CORE PASSPORT INFERENCE & DECOUPLED CATVTON TRY-ON HANDLERS (ZEROGPU)
# ============================================================================

def _estimate_run_gpu_duration(
    input_img=None,
    custom_color="#98f3ff",
    upscale_factor=4,
    eye_lock_strength=0.88,
    catchlight_boost=0.38,
    skin_smoothing=0.52,
    pore_detail_4k=0.65,
    relight_strength=0.55,
    dress_preset="none",
    dress_vertical_shift=0.0,
    dress_shoulder_scale=1.0,
    backdrop_style="solid",
    passport_standard="original_ratio",
    auto_level_head=False,
    candidate_name_stamp="",
    date_stamp="",
    add_photo_border=False,
    print_sheet_layout="single_photo",
    brightness=0.0,
    contrast=1.0,
    saturation=1.0,
    gamma=1.0,
    warmth=0.0,
    shadows_lift=0.0,
    use_ai_upscale=True,
    use_full_ai=True,
    dress_target_person="auto",
    custom_dress_img=None,
    compact_vton_state=None,
) -> int:
    """Dynamic ZeroGPU reservation (seconds) so H200 1.5x multiplier never over-reserves quota."""
    _ = (
        input_img, custom_color, upscale_factor, eye_lock_strength, catchlight_boost,
        skin_smoothing, pore_detail_4k, relight_strength, dress_vertical_shift,
        dress_shoulder_scale, backdrop_style, passport_standard, auto_level_head,
        candidate_name_stamp, date_stamp, add_photo_border, print_sheet_layout,
        brightness, contrast, saturation, gamma, warmth, shadows_lift,
        use_ai_upscale, use_full_ai,
    )
    dp = str(dress_preset or "none")
    wants_dress = (dp != "none") or (custom_dress_img is not None)
    if not wants_dress:
        return 14
    if isinstance(compact_vton_state, dict) and compact_vton_state.get("dressed_base_rgb") is not None:
        return 14
    if dp.startswith("couple_") or str(dress_target_person or "") == "both":
        return 30
    return 22


@spaces.GPU(duration=_estimate_run_gpu_duration)
def _gpu_generate_passport_v8(
    input_img: np.ndarray,
    custom_color: str = "#98f3ff",
    upscale_factor: int = 4,
    eye_lock_strength: float = 0.88,
    catchlight_boost: float = 0.38,
    skin_smoothing: float = 0.52,
    pore_detail_4k: float = 0.65,
    relight_strength: float = 0.55,
    dress_preset: str = "none",
    dress_vertical_shift: float = 0.0,
    dress_shoulder_scale: float = 1.0,
    backdrop_style: str = "solid",
    passport_standard: str = "original_ratio",
    auto_level_head: bool = False,
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
    use_ai_upscale: bool = True,
    use_full_ai: bool = True,
    dress_target_person: str = "auto",
    custom_dress_img: np.ndarray | None = None,
    compact_vton_state: dict | None = None,
):
    """ZeroGPU V8 worker returning full results + browser-session `session_state`."""
    if input_img is None:
        return None, None, None, None, None, {"error": "Please upload a portrait photo first."}, None

    norm_color = normalize_color_to_hex(custom_color, default="#98f3ff")
    result = process_photo_v3(
        input_img=input_img,
        custom_color=norm_color,
        upscale_factor=int(upscale_factor),
        eye_lock_strength=float(eye_lock_strength),
        catchlight_boost=float(catchlight_boost),
        skin_smoothing=float(skin_smoothing),
        pore_detail_4k=float(pore_detail_4k),
        relight_strength=float(relight_strength),
        dress_preset=str(dress_preset),
        dress_vertical_shift=float(dress_vertical_shift),
        dress_shoulder_scale=float(dress_shoulder_scale),
        backdrop_style=str(backdrop_style),
        passport_standard=str(passport_standard or "original_ratio"),
        auto_level_head=bool(auto_level_head),
        candidate_name_stamp=str(candidate_name_stamp or ""),
        date_stamp=str(date_stamp or ""),
        add_photo_border=bool(add_photo_border),
        print_sheet_layout=str(print_sheet_layout),
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        gamma=float(gamma),
        warmth=float(warmth),
        shadows_lift=float(shadows_lift),
        use_ai_upscale=bool(use_ai_upscale),
        use_full_ai=bool(use_full_ai),
        dress_target_person=str(dress_target_person or "auto"),
        custom_dress_img=custom_dress_img,
        prior_vton_state=compact_vton_state,
    )
    return (
        result["final_rgb"],
        result["sheet_rgb"],
        result["telemetry_rgb"],
        result["final_path"],
        result["sheet_path"],
        result["metrics"],
        result["session_state"],
    )


def gpu_generate_passport(
    input_img: np.ndarray,
    custom_color: str = "#98f3ff",
    upscale_factor: int = 4,
    eye_lock_strength: float = 0.88,
    catchlight_boost: float = 0.38,
    skin_smoothing: float = 0.52,
    pore_detail_4k: float = 0.65,
    relight_strength: float = 0.55,
    dress_preset: str = "none",
    dress_vertical_shift: float = 0.0,
    dress_shoulder_scale: float = 1.0,
    backdrop_style: str = "solid",
    passport_standard: str = "original_ratio",
    auto_level_head: bool = False,
    candidate_name_stamp: str = "",
    date_stamp: str = "",
    add_photo_border: bool = False,
    print_sheet_layout: str = "sheet_4x6_8up",
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
    use_ai_upscale: bool = True,
    use_full_ai: bool = True,
):
    """Backward-compatible 6-tuple passport generation handler."""
    final_rgb, sheet_rgb, telemetry_rgb, final_path, sheet_path, metrics, _sess = _gpu_generate_passport_v8(
        input_img=input_img,
        custom_color=custom_color,
        upscale_factor=upscale_factor,
        eye_lock_strength=eye_lock_strength,
        catchlight_boost=catchlight_boost,
        skin_smoothing=skin_smoothing,
        pore_detail_4k=pore_detail_4k,
        relight_strength=relight_strength,
        dress_preset=dress_preset,
        dress_vertical_shift=dress_vertical_shift,
        dress_shoulder_scale=dress_shoulder_scale,
        backdrop_style=backdrop_style,
        passport_standard=passport_standard,
        auto_level_head=auto_level_head,
        candidate_name_stamp=candidate_name_stamp,
        date_stamp=date_stamp,
        add_photo_border=add_photo_border,
        print_sheet_layout=print_sheet_layout,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        warmth=warmth,
        shadows_lift=shadows_lift,
        use_ai_upscale=use_ai_upscale,
        use_full_ai=use_full_ai,
    )
    return final_rgb, sheet_rgb, telemetry_rgb, final_path, sheet_path, metrics


def _estimate_tryon_gpu_duration(
    input_img=None,
    compact_state=None,
    dress_preset="none",
    dress_target_person="auto",
    custom_dress_img=None,
    dress_vertical_shift=0.0,
    dress_shoulder_scale=1.0,
    custom_color="#98f3ff",
    brightness=0.0,
    contrast=1.0,
    saturation=1.0,
    gamma=1.0,
    warmth=0.0,
    shadows_lift=0.0,
) -> int:
    _ = (
        input_img, compact_state, custom_dress_img, dress_vertical_shift,
        dress_shoulder_scale, custom_color, brightness, contrast,
        saturation, gamma, warmth, shadows_lift,
    )
    dp = str(dress_preset or "none")
    if dp.startswith("couple_") or str(dress_target_person or "") == "both":
        return 26
    return 18


@spaces.GPU(duration=_estimate_tryon_gpu_duration)
def _gpu_tryon_dress_v8(
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
):
    """ZeroGPU worker for decoupled AI Virtual Try-On (`CatVTON` — no repeated upscaling!)."""
    res = tryon_dress_decoupled(
        input_img=input_img,
        compact_state=compact_state,
        dress_preset=str(dress_preset or "none"),
        dress_target_person=str(dress_target_person or "auto"),
        custom_dress_img=custom_dress_img,
        dress_vertical_shift=float(dress_vertical_shift),
        dress_shoulder_scale=float(dress_shoulder_scale),
        custom_color=normalize_color_to_hex(custom_color, default="#98f3ff"),
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        gamma=float(gamma),
        warmth=float(warmth),
        shadows_lift=float(shadows_lift),
    )
    return res["final_rgb"], res["telemetry_rgb"], res["metrics"], res["session_state"]


def _ui_run_studio(
    input_img,
    bg_color_picker="#98f3ff",
    upscale_slider=4,
    auto_level_chk=False,
    use_ai_upscale_chk=True,
    use_full_ai_chk=True,
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    dress_dropdown="none",
    dress_vshift=0.0,
    dress_sh_scale=1.0,
    dress_target_dd="auto",
    custom_dress_upload=None,
    session_state=None,
    standard_dd="original_ratio",
    preview_mode_dd="single",
    stamp_name_txt="",
    stamp_date_txt="",
    add_border_chk=False,
):
    """
    Triggered ONLY when the user clicks the 'Run' button (or loads a saved customer).
    Processes exact uploaded framing (`original_ratio`), runs Multi-Subject AI + BRIA + CatVTON + 4K/8K SR,
    and updates browser `studio_session_state`.
    """
    if input_img is None:
        return None, None, {"status": "Upload a photo on the left, set your options, and click Run."}, None

    img_sig = compute_image_signature(input_img)
    compact_vton = extract_compact_vton_input_state(session_state, img_sig)
    norm_color = normalize_color_to_hex(bg_color_picker, default="#98f3ff")
    sheet_layout = "single_photo" if preview_mode_dd == "single" else str(preview_mode_dd)

    final_rgb, sheet_rgb, telemetry_rgb, _final_path, _sheet_path, metrics, new_sess = _gpu_generate_passport_v8(
        input_img=input_img,
        custom_color=norm_color,
        upscale_factor=int(upscale_slider),
        auto_level_head=bool(auto_level_chk),
        use_ai_upscale=bool(use_ai_upscale_chk),
        use_full_ai=bool(use_full_ai_chk),
        gamma=float(gamma_slider),
        brightness=float(bright_slider),
        contrast=float(contrast_slider),
        saturation=float(sat_slider),
        warmth=float(warmth_slider),
        shadows_lift=float(shadows_slider),
        dress_preset=str(dress_dropdown or "none"),
        dress_vertical_shift=float(dress_vshift),
        dress_shoulder_scale=float(dress_sh_scale),
        backdrop_style="solid",
        passport_standard=str(standard_dd or "original_ratio"),
        candidate_name_stamp=str(stamp_name_txt or ""),
        date_stamp=str(stamp_date_txt or ""),
        add_photo_border=bool(add_border_chk),
        print_sheet_layout=sheet_layout,
        dress_target_person=str(dress_target_dd or "auto"),
        custom_dress_img=custom_dress_upload,
        compact_vton_state=compact_vton,
    )

    live_preview = final_rgb if preview_mode_dd == "single" else sheet_rgb
    return live_preview, telemetry_rgb, metrics, new_sess


def _ui_tryon_dress(
    input_img,
    dress_dropdown="none",
    dress_target_dd="auto",
    custom_dress_upload=None,
    dress_vshift=0.0,
    dress_sh_scale=1.0,
    bg_color_picker="#98f3ff",
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    session_state=None,
):
    """
    Decoupled 'No-Run, Zero-Degradation' Dress Try-On Handler:
    When the user selects a dress or clicks `👗 Try On Dress`, runs ONLY AI Dress Try-On
    without requiring `🚀 Run` and NEVER stacks upscaling!
    """
    if input_img is None:
        noop = gr.update() if HAS_GRADIO else None
        return noop, noop, {"status": "Upload a photo on the left first to preview AI Dress Try-On."}, session_state

    # If user selected 'none' with no custom dress and hasn't run or dressed yet, don't waste GPU
    has_custom = custom_dress_upload is not None and isinstance(custom_dress_upload, np.ndarray)
    if (not dress_dropdown or str(dress_dropdown) == "none") and not has_custom and session_state is None:
        noop = gr.update() if HAS_GRADIO else None
        return noop, noop, gr.update() if HAS_GRADIO else {}, session_state

    img_sig = compute_image_signature(input_img)
    compact_state = extract_compact_vton_input_state(session_state, img_sig)
    norm_color = normalize_color_to_hex(bg_color_picker, default="#98f3ff")

    # Zero-GPU instant CPU restore when switching back to 'Keep Original Dress' with cached session!
    if (not dress_dropdown or str(dress_dropdown) == "none") and not has_custom and isinstance(compact_state, dict) and compact_state.get("clean_base_rgb") is not None:
        res_cpu = tryon_dress_decoupled(
            input_img=input_img,
            compact_state=compact_state,
            dress_preset="none",
            dress_target_person=str(dress_target_dd or "auto"),
            custom_dress_img=None,
            dress_vertical_shift=float(dress_vshift),
            dress_shoulder_scale=float(dress_sh_scale),
            custom_color=norm_color,
            brightness=float(bright_slider),
            contrast=float(contrast_slider),
            saturation=float(sat_slider),
            gamma=float(gamma_slider),
            warmth=float(warmth_slider),
            shadows_lift=float(shadows_slider),
        )
        return res_cpu["final_rgb"], res_cpu["telemetry_rgb"], res_cpu["metrics"], res_cpu["session_state"]

    preview_rgb, telemetry_rgb, metrics, new_sess = _gpu_tryon_dress_v8(
        input_img=input_img,
        compact_state=compact_state,
        dress_preset=str(dress_dropdown or "none"),
        dress_target_person=str(dress_target_dd or "auto"),
        custom_dress_img=custom_dress_upload,
        dress_vertical_shift=float(dress_vshift),
        dress_shoulder_scale=float(dress_sh_scale),
        custom_color=norm_color,
        brightness=float(bright_slider),
        contrast=float(contrast_slider),
        saturation=float(sat_slider),
        gamma=float(gamma_slider),
        warmth=float(warmth_slider),
        shadows_lift=float(shadows_slider),
    )
    return preview_rgb, telemetry_rgb, metrics, new_sess


def _ui_generate_passport(
    input_img,
    custom_color="#98f3ff",
    upscale_factor=3,
    eye_lock_strength=0.85,
    catchlight_boost=0.35,
    skin_smoothing=0.42,
    pore_detail_4k=0.55,
    relight_strength=0.35,
    dress_preset="none",
    dress_vertical_shift=0.0,
    dress_shoulder_scale=1.0,
    backdrop_style="solid",
    passport_standard="original_ratio",
    auto_level_head=False,
    candidate_name_stamp="",
    date_stamp="",
    add_photo_border=False,
    print_sheet_layout="sheet_4x6_8up",
    brightness=0.0,
    contrast=1.0,
    saturation=1.0,
    gamma=1.0,
    warmth=0.0,
    shadows_lift=0.0,
    use_ai_upscale=True,
    use_full_ai=True,
):
    """Backward-compatible full tuple UI wrapper."""
    if input_img is None:
        return (
            None,
            None,
            None,
            None,
            None,
            {},
            "👆 **Upload a photo** on the left and click **Run** to create your passport photo.",
        )

    norm_color = normalize_color_to_hex(custom_color, default="#98f3ff")
    final_rgb, sheet_rgb, telemetry_rgb, final_path, sheet_path, metrics = gpu_generate_passport(
        input_img=input_img,
        custom_color=norm_color,
        upscale_factor=upscale_factor,
        eye_lock_strength=eye_lock_strength,
        catchlight_boost=catchlight_boost,
        skin_smoothing=skin_smoothing,
        pore_detail_4k=pore_detail_4k,
        relight_strength=relight_strength,
        dress_preset=dress_preset,
        dress_vertical_shift=dress_vertical_shift,
        dress_shoulder_scale=dress_shoulder_scale,
        backdrop_style=backdrop_style,
        passport_standard=passport_standard,
        auto_level_head=auto_level_head,
        candidate_name_stamp=candidate_name_stamp,
        date_stamp=date_stamp,
        add_photo_border=add_photo_border,
        print_sheet_layout=print_sheet_layout,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gamma=gamma,
        warmth=warmth,
        shadows_lift=shadows_lift,
        use_ai_upscale=use_ai_upscale,
        use_full_ai=use_full_ai,
    )

    std_label = PASSPORT_STANDARDS.get(passport_standard, {}).get("label", "Original Ratio")
    dress_label = DRESS_PRESETS.get(dress_preset, {}).get("label", "Original Dress")
    status = f"✅ **Photo Ready!** ({std_label} • {dress_label})"
    if dress_preset == "none" and (abs(float(dress_vertical_shift)) > 0.01 or abs(float(dress_shoulder_scale) - 1.0) > 0.01):
        status += " — 💡 **Tip:** Select an outfit from **Select Dress** to apply shift/shoulder width."
    return final_rgb, sheet_rgb, telemetry_rgb, final_path, sheet_path, metrics, status


def _ui_live_manual_color(
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    bg_color_picker=None,
    session_state=None,
    input_img=None,
    preview_mode_dd="single",
    standard_dd="original_ratio",
    stamp_name_txt=None,
    stamp_date_txt=None,
    add_border_chk=None,
):
    """
    ZeroGPU-Safe (<20ms) Live Manual Colour Correction & Background Colour Re-Compositing:
    Works seamlessly both AFTER `Run` / `Try On` (via `session_state`) AND BEFORE `Run` (on `input_img`),
    keeping official studio backgrounds 100% pure and untinted!
    """
    _ = (preview_mode_dd, standard_dd, stamp_name_txt, stamp_date_txt, add_border_chk)
    updated_rgb = apply_live_manual_color_v8(
        session_state=session_state,
        input_img=input_img,
        brightness=float(bright_slider),
        contrast=float(contrast_slider),
        saturation=float(sat_slider),
        gamma=float(gamma_slider),
        warmth=float(warmth_slider),
        shadows_lift=float(shadows_slider),
        custom_color=bg_color_picker,
    )
    if updated_rgb is None:
        return gr.update() if HAS_GRADIO else None
    return updated_rgb


def _ui_reset_manual_color(
    bg_color_picker="#98f3ff",
    session_state=None,
    input_img=None,
):
    """Reset all 6 Manual Colour Correction sliders to neutral and refresh `Live Preview`."""
    refreshed = _ui_live_manual_color(
        gamma_slider=1.0,
        bright_slider=0.0,
        contrast_slider=1.0,
        sat_slider=1.0,
        warmth_slider=0.0,
        shadows_slider=0.0,
        bg_color_picker=bg_color_picker,
        session_state=session_state,
        input_img=input_img,
    )
    return 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, refreshed


def _on_bg_preset_change(
    preset_col,
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    session_state=None,
    input_img=None,
):
    """Sync preset dropdown to ColorPicker and live-update background immediately."""
    if str(preset_col or "").strip().lower() == "custom":
        noop = gr.update() if HAS_GRADIO else None
        return noop, noop
    hex_col = normalize_color_to_hex(preset_col, default="#98f3ff")
    # Only auto-render into Live Preview on bg change if session_state already exists (Run or Try-On done)
    if isinstance(session_state, dict) and session_state.get("fg_rgb") is not None:
        live_out = _ui_live_manual_color(
            gamma_slider=gamma_slider,
            bright_slider=bright_slider,
            contrast_slider=contrast_slider,
            sat_slider=sat_slider,
            warmth_slider=warmth_slider,
            shadows_slider=shadows_slider,
            bg_color_picker=hex_col,
            session_state=session_state,
            input_img=input_img,
        )
    else:
        live_out = gr.update() if HAS_GRADIO else None
    return hex_col, live_out


def _on_upload_sync_and_veil_check(
    input_img=None,
    dress_dropdown="none",
    dress_target_dd="auto",
    session_state=None,
):
    """
    Fast (<12ms CPU, Zero-GPU, No Auto-Run!) upload state sync & AI Veil (`Veli` / `Ghunghat`) inspection:
    - Caches `raw_upload_rgb` in `studio_session_state` so Manual Colour sliders never re-upload `input_image`.
    - Immediately alerts the studio operator if a female customer is wearing a head veil / high saree pallu!
    """
    if input_img is None or not isinstance(input_img, np.ndarray):
        return None, ""
    img_sig = compute_image_signature(input_img)
    preview_raw = smart_resize(input_img[:, :, :3], 1440)
    try:
        subjs = detect_foreground_subjects(preview_raw, raw_bria_alpha=None, max_subjects=4)
        veil_msg = build_veil_dress_reminder(
            subjs,
            dress_preset=str(dress_dropdown or "none"),
            target_person=str(dress_target_dd or "auto"),
        )
    except Exception:
        veil_msg = ""

    if isinstance(session_state, dict) and session_state.get("img_sig") == img_sig:
        session_state["raw_upload_rgb"] = preview_raw
        return session_state, veil_msg

    return {
        "img_sig": img_sig,
        "has_run_active": False,
        "raw_upload_rgb": preview_raw,
    }, veil_msg


def _ui_run_studio_gradio(
    input_img,
    bg_color_picker="#98f3ff",
    upscale_slider=4,
    auto_level_chk=False,
    use_ai_upscale_chk=True,
    use_full_ai_chk=True,
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    dress_dropdown="none",
    dress_vshift=0.0,
    dress_sh_scale=1.0,
    dress_target_dd="auto",
    custom_dress_upload=None,
    session_state=None,
):
    """Fast Unique-Filepath PNG UI wrapper for `🚀 Run` + AI Veil (`Veli`) Reminder Toast."""
    live_rgb, telem_rgb, metrics, new_sess = _ui_run_studio(
        input_img=input_img,
        bg_color_picker=bg_color_picker,
        upscale_slider=upscale_slider,
        auto_level_chk=auto_level_chk,
        use_ai_upscale_chk=use_ai_upscale_chk,
        use_full_ai_chk=use_full_ai_chk,
        gamma_slider=gamma_slider,
        bright_slider=bright_slider,
        contrast_slider=contrast_slider,
        sat_slider=sat_slider,
        warmth_slider=warmth_slider,
        shadows_slider=shadows_slider,
        dress_dropdown=dress_dropdown,
        dress_vshift=dress_vshift,
        dress_sh_scale=dress_sh_scale,
        dress_target_dd=dress_target_dd,
        custom_dress_upload=custom_dress_upload,
        session_state=session_state,
    )
    if live_rgb is None:
        return None, None, metrics, new_sess, ""
    if isinstance(new_sess, dict) and isinstance(input_img, np.ndarray):
        new_sess["raw_upload_rgb"] = smart_resize(input_img[:, :, :3], 1440)
    out_path = write_fast_unique_png(live_rgb, prefix="ms_v9_run")
    veil_msg = str(metrics.get("veil_reminder") or "") if isinstance(metrics, dict) else ""
    if veil_msg and str(dress_dropdown or "none") != "none" and HAS_GRADIO and hasattr(gr, "Warning"):
        gr.Warning(veil_msg.replace("**", "").replace("`", ""))
    return out_path, telem_rgb, metrics, new_sess, veil_msg


def _ui_tryon_dress_gradio(
    input_img,
    dress_dropdown="none",
    dress_target_dd="auto",
    custom_dress_upload=None,
    dress_vshift=0.0,
    dress_sh_scale=1.0,
    bg_color_picker="#98f3ff",
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    session_state=None,
):
    """Fast Unique-Filepath PNG UI wrapper for AI Virtual Try-On + AI Veil (`Veli`) Reminder Toast."""
    prev_rgb, telem_rgb, metrics, new_sess = _ui_tryon_dress(
        input_img=input_img,
        dress_dropdown=dress_dropdown,
        dress_target_dd=dress_target_dd,
        custom_dress_upload=custom_dress_upload,
        dress_vshift=dress_vshift,
        dress_sh_scale=dress_sh_scale,
        bg_color_picker=bg_color_picker,
        gamma_slider=gamma_slider,
        bright_slider=bright_slider,
        contrast_slider=contrast_slider,
        sat_slider=sat_slider,
        warmth_slider=warmth_slider,
        shadows_slider=shadows_slider,
        session_state=session_state,
    )
    if not isinstance(prev_rgb, np.ndarray):
        noop = gr.update() if HAS_GRADIO else None
        return prev_rgb, telem_rgb, metrics, new_sess, noop
    if isinstance(new_sess, dict) and isinstance(input_img, np.ndarray):
        new_sess["raw_upload_rgb"] = smart_resize(input_img[:, :, :3], 1440)
    out_path = write_fast_unique_png(prev_rgb, prefix="ms_v9_vton")
    veil_msg = str(metrics.get("veil_reminder") or "") if isinstance(metrics, dict) else ""
    if veil_msg and str(dress_dropdown or "none") != "none" and HAS_GRADIO and hasattr(gr, "Warning"):
        gr.Warning(veil_msg.replace("**", "").replace("`", ""))
    return out_path, telem_rgb, metrics, new_sess, veil_msg


def _ui_live_manual_color_gradio(
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    bg_color_picker=None,
    session_state=None,
):
    """
    Ultra-Fast (<85ms C++ SIMD PNG, ZERO `input_image` upload!) Manual Colour Commit:
    Writes a unique UUID `.png` file so Chrome repaints `✨ Live Preview` 100% deterministically
    on the very first frame without ever requiring a click on the result!
    """
    updated_rgb = apply_live_manual_color_v8(
        session_state=session_state,
        input_img=None,
        brightness=float(bright_slider),
        contrast=float(contrast_slider),
        saturation=float(sat_slider),
        gamma=float(gamma_slider),
        warmth=float(warmth_slider),
        shadows_lift=float(shadows_slider),
        custom_color=bg_color_picker,
    )
    if not isinstance(updated_rgb, np.ndarray):
        return gr.update() if HAS_GRADIO else None
    return write_fast_unique_png(updated_rgb, prefix="ms_v9_col")


def _ui_reset_manual_color_gradio(
    bg_color_picker="#98f3ff",
    session_state=None,
):
    """Reset all 6 Manual Colour sliders to neutral and return a fresh lossless PNG path."""
    refreshed_path = _ui_live_manual_color_gradio(
        gamma_slider=1.0,
        bright_slider=0.0,
        contrast_slider=1.0,
        sat_slider=1.0,
        warmth_slider=0.0,
        shadows_slider=0.0,
        bg_color_picker=bg_color_picker,
        session_state=session_state,
    )
    return 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, refreshed_path


def _on_bg_preset_change_gradio(
    preset_col,
    gamma_slider=1.0,
    bright_slider=0.0,
    contrast_slider=1.0,
    sat_slider=1.0,
    warmth_slider=0.0,
    shadows_slider=0.0,
    session_state=None,
):
    """Sync Background Colour dropdown to Colour Picker and immediately commit fresh lossless PNG."""
    if str(preset_col or "").strip().lower() == "custom":
        noop = gr.update() if HAS_GRADIO else None
        return noop, noop
    hex_col = normalize_color_to_hex(preset_col, default="#98f3ff")
    if isinstance(session_state, dict) and session_state.get("fg_rgb") is not None:
        live_out = _ui_live_manual_color_gradio(
            gamma_slider=gamma_slider,
            bright_slider=bright_slider,
            contrast_slider=contrast_slider,
            sat_slider=sat_slider,
            warmth_slider=warmth_slider,
            shadows_slider=shadows_slider,
            bg_color_picker=hex_col,
            session_state=session_state,
        )
    else:
        live_out = gr.update() if HAS_GRADIO else None
    return hex_col, live_out


# ============================================================================
# 2. DATABASE, GALLERY & SYSTEM HANDLERS
# ============================================================================

def handle_save_customer(
    name: str,
    input_img: np.ndarray | None,
    custom_color: str = "#98f3ff",
    upscale_factor: int = 4,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    dress_preset: str = "none",
    eye_lock_strength: float = 0.85,
    pore_detail_4k: float = 0.55,
    relight_strength: float = 0.35,
    passport_standard: str = "original_ratio",
    update_cust_id: str = "",
    catchlight_boost: float = 0.35,
    skin_smoothing: float = 0.42,
    dress_vertical_shift: float = 0.0,
    dress_shoulder_scale: float = 1.0,
    backdrop_style: str = "solid",
    print_sheet_layout: str = "sheet_4x6_8up",
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
    auto_level_head: bool = False,
    use_ai_upscale: bool = True,
    use_full_ai: bool = True,
    dress_target_person: str = "auto",
):
    """Save customer record and settings into SQLite and sync to Hugging Face Dataset."""
    if input_img is None:
        return "❌ Please upload a photo first.", update_cust_id
    norm_color = normalize_color_to_hex(custom_color, default="#98f3ff")
    extra = {
        "dress_preset": dress_preset,
        "dress_target_person": str(dress_target_person or "auto"),
        "eye_lock_strength": float(eye_lock_strength),
        "pore_detail_4k": float(pore_detail_4k),
        "relight_strength": float(relight_strength),
        "passport_standard": passport_standard,
        "catchlight_boost": float(catchlight_boost),
        "skin_smoothing": float(skin_smoothing),
        "dress_vertical_shift": float(dress_vertical_shift),
        "dress_shoulder_scale": float(dress_shoulder_scale),
        "backdrop_style": str(backdrop_style),
        "print_sheet_layout": str(print_sheet_layout),
        "warmth": float(warmth),
        "shadows_lift": float(shadows_lift),
        "auto_level_head": bool(auto_level_head),
        "use_ai_upscale": bool(use_ai_upscale),
        "use_full_ai": bool(use_full_ai),
    }
    msg, cid = save_customer_recipe(
        name=name,
        input_img=input_img,
        custom_color=norm_color,
        upscale_factor=int(upscale_factor),
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        gamma=float(gamma),
        update_cust_id=update_cust_id.strip() or None,
        extra_recipe=extra,
    )
    return msg, (cid or update_cust_id)


def _ui_save_customer_v4(
    name: str,
    input_img: np.ndarray | None,
    custom_color: str,
    upscale_factor: int,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma: float,
    dress_preset: str,
    update_cust_id: str = "",
    warmth: float = 0.0,
    shadows_lift: float = 0.0,
    auto_level_head: bool = False,
    use_ai_upscale: bool = True,
    use_full_ai: bool = True,
    dress_vertical_shift: float = 0.0,
    dress_shoulder_scale: float = 1.0,
    dress_target_person: str = "auto",
    passport_standard: str = "original_ratio",
):
    msg, cid = handle_save_customer(
        name=name,
        input_img=input_img,
        custom_color=normalize_color_to_hex(custom_color, default="#98f3ff"),
        upscale_factor=int(upscale_factor),
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        gamma=float(gamma),
        dress_preset=dress_preset,
        passport_standard=passport_standard,
        update_cust_id=update_cust_id,
        warmth=float(warmth),
        shadows_lift=float(shadows_lift),
        auto_level_head=bool(auto_level_head),
        use_ai_upscale=bool(use_ai_upscale),
        use_full_ai=bool(use_full_ai),
        dress_vertical_shift=float(dress_vertical_shift),
        dress_shoulder_scale=float(dress_shoulder_scale),
        dress_target_person=str(dress_target_person or "auto"),
    )
    gallery_items, table_rows = handle_search_customers("")
    return msg, cid, gallery_items, table_rows


def handle_search_customers(query: str = ""):
    """Return gallery items and enriched 4-column table rows matching customer search query."""
    detailed = search_customers_detailed(query)
    gallery_items = []
    table_rows = []
    for item in detailed:
        cid = item["cust_id"]
        cname = item["name"]
        cdate = item["date"]
        csum = item["summary"]
        tpath = item["thumbnail_path"]
        if tpath and os.path.exists(tpath):
            gallery_items.append((tpath, f"{cid}: {cname}"))
        table_rows.append([cid, cname, cdate, csum])
    return gallery_items, table_rows


def handle_load_customer(cust_id: str):
    """Load customer photo and saved settings (backward-compatible tuple)."""
    cid = (cust_id or "").strip()
    if ":" in cid:
        cid = cid.split(":", 1)[0].strip()
    if not cid:
        return (
            None, "", "", "#98f3ff", 4, 0.0, 1.0, 1.0, 1.0, "none",
            0.85, 0.55, 0.35, "original_ratio",
            "❌ Enter a valid Customer ID (e.g. ms-1)",
            0.35, 0.42, 0.0, 1.0, "solid", "sheet_4x6_8up",
        )
    img, recipe = load_permanent_customer_recipe(cid)
    if img is None or recipe is None:
        return (
            None, cid, "", "#98f3ff", 4, 0.0, 1.0, 1.0, 1.0, "none",
            0.85, 0.55, 0.35, "original_ratio",
            f"❌ Customer {cid} not found",
            0.35, 0.42, 0.0, 1.0, "solid", "sheet_4x6_8up",
        )
    return (
        img,
        cid,
        recipe.get("name", ""),
        normalize_color_to_hex(recipe.get("color", "#98f3ff"), default="#98f3ff"),
        int(recipe.get("upscale", 4)),
        float(recipe.get("brightness", 0.0)),
        float(recipe.get("contrast", 1.0)),
        float(recipe.get("saturation", 1.0)),
        float(recipe.get("gamma", 1.0)),
        recipe.get("dress_preset", "none"),
        float(recipe.get("eye_lock_strength", 0.85)),
        float(recipe.get("pore_detail_4k", 0.55)),
        float(recipe.get("relight_strength", 0.35)),
        recipe.get("passport_standard", "original_ratio"),
        f"✅ Loaded {cid} ({recipe.get('name', '')})",
        float(recipe.get("catchlight_boost", 0.35)),
        float(recipe.get("skin_smoothing", 0.42)),
        float(recipe.get("dress_vertical_shift", 0.0)),
        float(recipe.get("dress_shoulder_scale", 1.0)),
        recipe.get("backdrop_style", "solid"),
        recipe.get("print_sheet_layout", "sheet_4x6_8up"),
    )


def _ui_load_customer_v4(cust_id: str):
    """Load customer photo and saved settings back into V8 studio controls."""
    cid = (cust_id or "").strip()
    if ":" in cid:
        cid = cid.split(":", 1)[0].strip()
    if not cid:
        return (
            None, None, "", "", "#98f3ff", "#98f3ff", 4, False, True, True,
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            "none", "auto", None, 0.0, 1.0,
            "❌ Select a customer from the table/gallery or enter an ID (e.g. ms-1)",
        )
    img, recipe = load_permanent_customer_recipe(cid)
    if img is None or recipe is None:
        return (
            None, None, cid, "", "#98f3ff", "#98f3ff", 4, False, True, True,
            1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            "none", "auto", None, 0.0, 1.0,
            f"❌ Customer {cid} not found",
        )
    col = normalize_color_to_hex(recipe.get("color", "#98f3ff"), default="#98f3ff")
    preset_colors = {"#98f3ff", "#ffffff", "#1d4ed8", "#e2e8f0", "#fef9c3", "#dc2626"}
    dd_col = col if col in preset_colors else "custom"
    return (
        img,
        None,
        cid,
        recipe.get("name", ""),
        dd_col,
        col,
        int(recipe.get("upscale", 4)),
        bool(recipe.get("auto_level_head", False)),
        bool(recipe.get("use_ai_upscale", True)),
        bool(recipe.get("use_full_ai", True)),
        float(recipe.get("gamma", 1.0)),
        float(recipe.get("brightness", 0.0)),
        float(recipe.get("contrast", 1.0)),
        float(recipe.get("saturation", 1.0)),
        float(recipe.get("warmth", 0.0)),
        float(recipe.get("shadows_lift", 0.0)),
        recipe.get("dress_preset", "none"),
        recipe.get("dress_target_person", "auto"),
        None,
        float(recipe.get("dress_vertical_shift", 0.0)),
        float(recipe.get("dress_shoulder_scale", 1.0)),
        f"✅ Loaded {cid}: {recipe.get('name', '')} — Ready in Photo Studio!",
    )


def _on_table_select(table_data, evt: "gr.SelectData"):
    """Extract customer ID when user clicks any row in the Saved Customer table."""
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
        if hasattr(table_data, "iloc"):
            cid = str(table_data.iloc[row_idx, 0])
        else:
            cid = str(table_data[row_idx][0])
        return cid.strip()
    except Exception:
        return ""


def _on_gallery_select(evt: "gr.SelectData"):
    """Extract customer ID when user clicks any thumbnail in the Saved Customer gallery."""
    try:
        val = evt.value
        if isinstance(val, dict):
            cap = str(val.get("caption", ""))
        elif isinstance(val, (list, tuple)) and len(val) > 1:
            cap = str(val[1])
        else:
            cap = str(val)
        if ":" in cap:
            return cap.split(":", 1)[0].strip()
        return cap.strip()
    except Exception:
        return ""


def handle_delete_customer(cust_id: str):
    cid = (cust_id or "").strip()
    if ":" in cid:
        cid = cid.split(":", 1)[0].strip()
    if not cid:
        g, r = handle_search_customers("")
        return "❌ Select or enter a Customer ID to delete.", g, r
    msg = delete_customer(cid)
    g, r = handle_search_customers("")
    return msg, g, r


def handle_refresh_system_status():
    size_mb, pct, last_backup, _, date_range = get_system_status()
    all_cust = get_all_gallery()
    return {
        "Total Saved Customers": len(all_cust),
        "Storage Used (MB)": round(size_mb, 2),
        "Storage Limit (MB)": 3000.0,
        "Storage Used (%)": f"{round(pct, 1)}%",
        "Records Date Range": date_range,
        "Last Backup": last_backup,
    }


def _ui_open_mobile_qr_shoot(request: "gr.Request | None" = None):
    """Generate a live Zero-HF-Login Phone Camera QR Shoot session."""
    sess = create_mobile_shoot_session(request=request)
    msg = (
        f"📱 **Studio Camera Pro V8 (Zero HF Login, Zero-413 Chunked HD/PNG):**\n"
        f"[{sess['shoot_url']}]({sess['shoot_url']})\n\n"
        f"Includes **👤 Single / 👥 Couple / 🪪 Half-Body** guides, **1.4× Portrait** zoom, **✨ 3-Frame Anti-Shake**, and **Continuous Multi-Customer Shoot** — photos appear in **📷 Upload Photo** automatically!"
    )
    return sess["session_id"], sess["qr_rgb"], msg


def _ui_load_mobile_shoot_photo(session_id: str = ""):
    """Load the photo captured on the phone into the Studio's Upload Photo canvas."""
    img, msg = poll_mobile_shoot_photo_status(session_id, only_new=False)
    if img is None:
        return (gr.update() if HAS_GRADIO else None), msg
    return img, msg


def _ui_auto_poll_mobile_shoot(session_id: str = ""):
    """Background 1.8s Timer poll: auto-populates Upload Photo when a newly uploaded phone photo arrives."""
    img, msg = poll_mobile_shoot_photo_status(session_id, only_new=True)
    if img is None:
        return (gr.update() if HAS_GRADIO else None), (gr.update() if HAS_GRADIO else "")
    return img, msg


def _ui_submit_mobile_camera_photo(session_id: str, mobile_img: np.ndarray | None):
    """Receive photo from mobile phone (?mobile_shoot=<session_id>) and store it for the Studio."""
    if mobile_img is None:
        return "❌ Please take a photo with your phone camera or select an image first."
    saved_path = save_mobile_shoot_photo(session_id or "mobile", mobile_img)
    return (
        f"✅ **Photo Sent to Studio!** (`{os.path.basename(saved_path)}`)\n\n"
        f"It will automatically appear in **📷 Upload Photo** on your Studio screen (or click **📥 Load Photo from Phone**)!"
    )


def _ui_create_customer_share(
    output_img: np.ndarray | str | None,
    cust_name: str = "",
    pin_input: str = "",
    session_state: dict | None = None,
    request: "gr.Request | None" = None,
):
    """Create a permanent, 4-Digit PIN protected customer share link (?share=<share_id>) & QR code."""
    target_img: np.ndarray | None = None
    if isinstance(output_img, np.ndarray):
        target_img = output_img
    elif isinstance(output_img, str) and os.path.isfile(output_img):
        target_img = imread_rgb(output_img)
    if target_img is None and isinstance(session_state, dict):
        fg = session_state.get("fg_rgb")
        if isinstance(fg, np.ndarray):
            target_img = apply_live_manual_color_v8(session_state=session_state)
    if target_img is None:
        cached = _LAST_RUN_CACHE.get("base_composited")
        if isinstance(cached, np.ndarray):
            target_img = cached
    if target_img is None:
        return "", "", None, "❌ Please click **🚀 Run** or **👗 Try On Dress** first before creating a share link."

    share = create_customer_share(
        photo_rgb=target_img,
        customer_name=cust_name or "Studio Customer",
        pin_code=pin_input,
        request=request,
    )
    md = (
        f"✅ **Zero-HF-Login Cloud Share Vault Created!**\n"
        f"- **Customer Share URL (No Login Needed):** [{share['share_url']}]({share['share_url']})\n"
        f"- **4-Digit Security PIN:** `{share['pin_code']}`\n"
        f"- **Full Resolution:** `{share['resolution_str']}` (1x–8x Deep Zoom + Lossless PNG Download)"
    )
    return share["share_url"], share["pin_code"], share["qr_rgb"], md


def _ui_unlock_customer_share(share_id: str, entered_pin: str):
    """Verify 4-Digit PIN and render 1x–8x Deep-Zoom HTML5 viewer + lossless PNG file download."""
    ok, _photo_rgb, photo_path, meta = verify_customer_share_pin(share_id, entered_pin)
    if not ok or meta is None:
        msg = meta.get("message", "❌ Incorrect 4-Digit PIN.") if isinstance(meta, dict) else "❌ Incorrect 4-Digit PIN."
        return msg, "", None
    return meta.get("message", "✅ PIN Verified!"), meta.get("deep_zoom_html", ""), photo_path


def _ui_route_on_load(request: "gr.Request | None" = None):
    """Route URL query params (?mobile_shoot=... / ?share=...) or show the 4-Tab Photo Studio."""
    q_mobile = ""
    q_share = ""
    if request is not None:
        try:
            qp = getattr(request, "query_params", {}) or {}
            q_mobile = str(qp.get("mobile_shoot", "") or "").strip()
            q_share = str(qp.get("share", "") or "").strip()
        except Exception:
            pass

    sess = create_mobile_shoot_session(request=request)
    default_qr_msg = (
        f"📱 **Scan on Any Phone (Zero HF Login):** [{sess['shoot_url']}]({sess['shoot_url']})"
    )

    if q_mobile:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            q_mobile,
            f"### 📱 Mobile Camera Shoot Connected (`{q_mobile}`)\nTake a portrait photo below and tap **📤 Send Photo to Studio**!",
            "",
            "",
            sess["session_id"],
            sess["qr_rgb"],
            default_qr_msg,
        )

    if q_share:
        meta = get_customer_share_meta(q_share)
        if meta is not None:
            hdr = (
                f"### 🔒 Customer Photo Vault — **{meta['customer_name']}** (`{meta['resolution_str']}`)\n"
                f"Enter your **4-Digit Security PIN** below to unlock 1x–8x Deep-Zoom inspection & original PNG download."
            )
        else:
            hdr = f"### 🔒 Customer Photo Vault (`{q_share}`)\nEnter your **4-Digit Security PIN** below to unlock your photo."
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            "",
            q_share,
            hdr,
            sess["session_id"],
            sess["qr_rgb"],
            default_qr_msg,
        )

    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        sess["session_id"],
        "",
        "",
        "",
        sess["session_id"],
        sess["qr_rgb"],
        default_qr_msg,
    )


# ============================================================================
# 3. CLEAN, NON-SCROLLABLE LEFT + SCROLLABLE RIGHT STUDIO INTERFACE (V8)
# ============================================================================

STUDIO_CSS = """
/* Top simple studio bar */
.studio-top-title {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: #f8fafc;
    padding: 8px 18px;
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.studio-top-title h1 {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #ffffff;
}

/* Desktop split-screen lock: Left side NEVER scrolls; ONLY Right side scrolls */
@media (min-width: 768px) {
    html, body {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        margin: 0 !important;
    }
    .gradio-container {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        padding: 8px 14px !important;
        max-width: 100% !important;
    }
    .studio-workspace-row {
        height: calc(100vh - 96px) !important;
        max-height: calc(100vh - 96px) !important;
        overflow: hidden !important;
        align-items: stretch !important;
        gap: 14px !important;
    }
    .studio-left-locked {
        position: sticky !important;
        top: 0 !important;
        height: calc(100vh - 96px) !important;
        max-height: calc(100vh - 96px) !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
    }
    .studio-two-screens-row {
        height: 100% !important;
        max-height: calc(100vh - 96px) !important;
        overflow: hidden !important;
        align-items: stretch !important;
        gap: 12px !important;
    }
    .studio-right-scroll {
        height: calc(100vh - 96px) !important;
        max-height: calc(100vh - 96px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 8px !important;
        scrollbar-width: thin;
    }
    .studio-tab-scrollable {
        height: calc(100vh - 96px) !important;
        max-height: calc(100vh - 96px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 6px !important;
    }
}
"""


STUDIO_CLIENT_JS = """
(function() {
    if (window.__msStudioClientInit) return;
    window.__msStudioClientInit = true;

    document.addEventListener('click', function(e) {
        var btn = e.target && e.target.closest ? e.target.closest('button') : null;
        if (!btn) return;
        var upBox = document.getElementById('studio_upload_image');
        if (upBox && upBox.contains(btn)) {
            var lbl = ((btn.getAttribute('aria-label') || '') + ' ' + (btn.getAttribute('title') || '')).toLowerCase();
            if (lbl.indexOf('webcam') !== -1 || lbl.indexOf('camera') !== -1) {
                var qrBtn = document.getElementById('phone_qr_shoot_btn');
                if (qrBtn) { qrBtn.click(); }
                var acc = document.getElementById('phone_qr_accordion');
                if (acc) {
                    var toggleBtn = acc.querySelector('button');
                    if (toggleBtn && (acc.querySelector('.hide') || toggleBtn.getAttribute('aria-expanded') === 'false')) {
                        toggleBtn.click();
                    }
                    acc.scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            }
        }
    }, true);

    // --- V9/V10 Zero-Latency 60-FPS Client-Side Foreground-Scoped Manual Colour Preview ---
    var baseSrc = '';
    var srcPixels = null, srcW = 0, srcH = 0;
    var baseP = {g: 1.0, b: 0.0, c: 1.0, s: 1.0, w: 0.0, sh: 0.0, bg: [152, 243, 255]};
    var rafId = 0, numTimer = 0;

    function readVal(id, defV) {
        var el = document.querySelector('#' + id + ' input[type="range"]') || document.querySelector('#' + id + ' input');
        if (!el) return defV;
        var v = parseFloat(el.value);
        return isNaN(v) ? defV : v;
    }
    function parseHexRgb(str) {
        if (!str) return [152, 243, 255];
        var m = String(str).trim().match(/^#?([0-9a-f]{6})/i);
        if (m) {
            var h = m[1];
            return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
        }
        return [152, 243, 255];
    }
    function readBgRgb() {
        var el = document.querySelector('#mc_bg_picker input[type="color"]') || document.querySelector('#mc_bg_picker input');
        return el ? parseHexRgb(el.value) : [152, 243, 255];
    }
    function readAll() {
        return {
            g: readVal('mc_gamma', 1.0),
            b: readVal('mc_bright', 0.0),
            c: readVal('mc_contrast', 1.0),
            s: readVal('mc_sat', 1.0),
            w: readVal('mc_warmth', 0.0),
            sh: readVal('mc_shadows', 0.0),
            bg: readBgRgb()
        };
    }
    function captureImgIfNeeded(img) {
        if (!img || !img.complete || !img.naturalWidth) return false;
        if (img.src === baseSrc && srcPixels) return true;
        try {
            var maxD = 820;
            var sc = Math.min(1.0, maxD / Math.max(img.naturalWidth, img.naturalHeight));
            srcW = Math.max(16, Math.round(img.naturalWidth * sc));
            srcH = Math.max(16, Math.round(img.naturalHeight * sc));
            var off = document.createElement('canvas');
            off.width = srcW; off.height = srcH;
            var ctx = off.getContext('2d', {willReadFrequently: true});
            ctx.drawImage(img, 0, 0, srcW, srcH);
            srcPixels = ctx.getImageData(0, 0, srcW, srcH).data;
            baseSrc = img.src;
            baseP.bg = [srcPixels[0], srcPixels[1], srcPixels[2]];
            return true;
        } catch (err) {
            return false;
        }
    }
    function ensureOverlayCanvas(img) {
        var parent = img.parentElement;
        if (!parent) return null;
        if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
        var cv = parent.querySelector('canvas.ms-v9-live-canvas');
        if (!cv) {
            cv = document.createElement('canvas');
            cv.className = 'ms-v9-live-canvas';
            cv.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;z-index:4;display:none;';
            parent.appendChild(cv);
        }
        return cv;
    }
    function renderLiveFrame() {
        rafId = 0;
        var box = document.getElementById('studio_live_preview');
        var img = box ? box.querySelector('img') : null;
        if (!img || !captureImgIfNeeded(img)) return;
        var cur = readAll();
        var cv = ensureOverlayCanvas(img);
        if (!cv) return;
        if (cv.width !== srcW || cv.height !== srcH) { cv.width = srcW; cv.height = srcH; }

        var invG = Math.max(0.1, baseP.g) / Math.max(0.1, cur.g);
        var dC = cur.c / Math.max(0.2, baseP.c);
        var dB = cur.b - baseP.b;
        var dS = cur.s / Math.max(0.12, baseP.s);
        var dW = cur.w - baseP.w;
        var dSh = cur.sh - baseP.sh;

        var lutR = new Uint8ClampedArray(256);
        var lutG = new Uint8ClampedArray(256);
        var lutB = new Uint8ClampedArray(256);
        for (var i = 0; i < 256; i++) {
            var x = Math.pow(i / 255.0, invG) * 255.0;
            if (Math.abs(dSh) > 0.05) {
                var nx = x / 255.0;
                var env = Math.pow(1.0 - nx, 1.75) * (0.22 + 3.25 * nx);
                x = Math.min(255.0, Math.max(0.0, x + dSh * 1.35 * env));
            }
            x = Math.min(255.0, Math.max(0.0, (x - 128.0) * dC + 128.0 + dB));
            lutR[i] = x + 0.68 * dW;
            lutG[i] = x + 0.14 * dW;
            lutB[i] = x - 0.72 * dW;
        }

        var ctx = cv.getContext('2d');
        var outImg = ctx.createImageData(srcW, srcH);
        var dst = outImg.data;
        var bgR0 = baseP.bg[0], bgG0 = baseP.bg[1], bgB0 = baseP.bg[2];
        var bgR1 = cur.bg[0], bgG1 = cur.bg[1], bgB1 = cur.bg[2];
        var hasSat = Math.abs(dS - 1.0) > 0.008;

        for (var p = 0, n = srcPixels.length; p < n; p += 4) {
            var r0 = srcPixels[p], g0 = srcPixels[p+1], b0 = srcPixels[p+2];
            var dr = r0 - bgR0, dg = g0 - bgG0, db = b0 - bgB0;
            var dist2 = dr*dr + dg*dg + db*db;
            if (dist2 <= 64) {
                dst[p] = bgR1; dst[p+1] = bgG1; dst[p+2] = bgB1; dst[p+3] = 255;
            } else {
                var rr = lutR[r0], gg = lutG[g0], bb = lutB[b0];
                if (hasSat) {
                    var lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb;
                    rr = lum + (rr - lum) * dS;
                    gg = lum + (gg - lum) * dS;
                    bb = lum + (bb - lum) * dS;
                }
                if (dist2 < 484) {
                    var w = (Math.sqrt(dist2) - 8.0) / 14.0;
                    rr = bgR1 + (rr - bgR1) * w;
                    gg = bgG1 + (gg - bgG1) * w;
                    bb = bgB1 + (bb - bgB1) * w;
                }
                dst[p] = rr; dst[p+1] = gg; dst[p+2] = bb; dst[p+3] = 255;
            }
        }
        ctx.putImageData(outImg, 0, 0);
        cv.style.display = 'block';
    }

    document.addEventListener('input', function(e) {
        var t = e.target;
        if (!t || !t.closest) return;
        if (t.closest('#manual_color_accordion') || t.closest('#mc_bg_picker')) {
            if (!rafId) rafId = requestAnimationFrame(renderLiveFrame);
            if (t.type === 'number') {
                clearTimeout(numTimer);
                numTimer = setTimeout(function() {
                    var cb = document.getElementById('studio_color_commit_btn');
                    if (cb) cb.click();
                }, 220);
            }
        }
    }, true);

    document.addEventListener('click', function(e) {
        var btn = e.target && e.target.closest ? e.target.closest('#manual_color_accordion button') : null;
        if (btn && btn.id !== 'mc_reset_all_btn' && btn.id !== 'studio_color_commit_btn') {
            setTimeout(function() {
                var cb = document.getElementById('studio_color_commit_btn');
                if (cb) cb.click();
            }, 60);
        }
    }, true);

    document.addEventListener('load', function(e) {
        var t = e.target;
        if (t && t.tagName === 'IMG' && t.closest && t.closest('#studio_live_preview')) {
            baseSrc = '';
            srcPixels = null;
            baseP = readAll();
            var cv = t.parentElement ? t.parentElement.querySelector('canvas.ms-v9-live-canvas') : null;
            if (cv) cv.style.display = 'none';
        }
    }, true);
})();
"""

STUDIO_HEAD_HTML = f"<style>{STUDIO_CSS}</style>\n<script>{STUDIO_CLIENT_JS}</script>"


def build_gradio_app():
    """Construct the Maa Sitala Passport Photo Studio Gradio 6.32+ / 5.x Blocks interface."""
    if not HAS_GRADIO:
        raise RuntimeError("Gradio is not installed in this Python environment.")

    dress_choices = [(v["label"], k) for k, v in DRESS_PRESETS.items()]
    dress_target_choices = [
        ("🤖 Auto AI Match (Smart Gender / Couple)", "auto"),
        ("⬅️ Left Person (Person 1)", "person_1"),
        ("➡️ Right Person (Person 2)", "person_2"),
        ("👥 Both People", "both"),
    ]
    bg_preset_choices = [
        ("🩵 Light Blue (Default)", "#98f3ff"),
        ("⬜ Pure White", "#ffffff"),
        ("🟦 Royal Blue", "#1d4ed8"),
        ("🩶 Light Gray", "#e2e8f0"),
        ("💛 Soft Cream", "#fef9c3"),
        ("❤️ Crimson Red", "#dc2626"),
        ("🎨 Custom Colour (Picker)", "custom"),
    ]

    # Configure prebuilt Gradio 6 (`buttons=["download", "fullscreen"]`) / Gradio 5 (`show_download_button=True`)
    # lossless PNG preview & download (`type="filepath"` + fast C++ `IMWRITE_PNG_COMPRESSION=1` unique UUID `.png`)
    _img_sig = inspect.signature(gr.Image.__init__).parameters
    _out_img_kwargs = {
        "label": "✨ Live Preview",
        "type": "filepath",
        "height": 440,
        "interactive": False,
        "elem_id": "studio_live_preview",
    }
    if "format" in _img_sig:
        _out_img_kwargs["format"] = "png"
    if "buttons" in _img_sig:
        _out_img_kwargs["buttons"] = ["download", "fullscreen"]
    elif "show_download_button" in _img_sig:
        _out_img_kwargs["show_download_button"] = True

    _blocks_sig = inspect.signature(gr.Blocks.__init__).parameters
    _blocks_kwargs: dict[str, object] = {"title": "Maa Sitala Passport Photo Studio"}
    if "css" in _blocks_sig:
        _blocks_kwargs["css"] = STUDIO_CSS
    if "head" in _blocks_sig:
        _blocks_kwargs["head"] = STUDIO_HEAD_HTML

    _html_sig = inspect.signature(gr.HTML.__init__).parameters
    _top_html_kwargs: dict[str, object] = {}
    if "head" in _html_sig:
        _top_html_kwargs["head"] = STUDIO_HEAD_HTML

    with gr.Blocks(**_blocks_kwargs) as demo:
        # Browser-session state holding processed foreground, alpha & pristine un-upscaled base for ZeroGPU
        studio_session_state = gr.State(None)

        # 1. SIMPLE PAGE TOP HEADER + GRADIO 6/5 CSS & CLIENT LIVE COLOUR ENGINE
        gr.HTML(
            f'<style>{STUDIO_CSS}</style><div class="studio-top-title"><h1>Maa Sitala Passport Photo Studio</h1></div>',
            **_top_html_kwargs,
        )

        # ====================================================================
        # PORTAL A: MOBILE PHONE CAMERA SHOOT VIEW (?mobile_shoot=<session_id>)
        # ====================================================================
        with gr.Group(visible=False) as mobile_shoot_portal_group:
            mobile_shoot_header_md = gr.Markdown("### 📱 Mobile Phone Camera Shoot")
            with gr.Row():
                mobile_shoot_sid_box = gr.Textbox(
                    label="Studio Session ID",
                    interactive=True,
                )
                back_to_studio_from_mobile_btn = gr.Button("🖥️ Switch to Full Studio View", variant="secondary")
            mobile_camera_input = gr.Image(
                label="📸 Snap or Select Portrait Photo on Phone",
                type="numpy",
                image_mode="RGB",
                height=380,
            )
            mobile_send_btn = gr.Button("📤 Send Photo to Studio", variant="primary", size="lg")
            mobile_send_status_md = gr.Markdown("")

        # ====================================================================
        # PORTAL B: CUSTOMER 4-DIGIT PIN SHARE VAULT (?share=<share_id>)
        # ====================================================================
        with gr.Group(visible=False) as share_vault_portal_group:
            vault_header_md = gr.Markdown("### 🔒 Customer Photo Share Vault")
            with gr.Row():
                vault_share_id_box = gr.Textbox(
                    label="Share ID",
                    interactive=True,
                    scale=2,
                )
                vault_pin_box = gr.Textbox(
                    label="🔑 Enter 4-Digit Security PIN",
                    placeholder="e.g. 4508",
                    type="password",
                    max_lines=1,
                    scale=2,
                )
                vault_unlock_btn = gr.Button("🔓 Unlock & Inspect 8x HD Photo", variant="primary", scale=2)
                back_to_studio_from_vault_btn = gr.Button("🖥️ Studio Home", variant="secondary", scale=1)
            vault_status_md = gr.Markdown("")
            vault_deep_zoom_html = gr.HTML("")
            vault_download_file = gr.File(label="📥 Lossless Original PNG File Download")

        # ====================================================================
        # MAIN 4-TAB STUDIO WORKSPACE
        # ====================================================================
        with gr.Group(visible=True) as main_studio_tabs_group:
            # 2. EXACT 4 TABS: "Photo Studio", "Saved Customer", "Backup", "Face Guide"
            with gr.Tabs():
                # ================================================================
                # TAB 1: PHOTO STUDIO (NON-SCROLLABLE LEFT + SCROLLABLE RIGHT)
                # ================================================================
                with gr.TabItem("Photo Studio"):
                    with gr.Row(elem_classes=["studio-workspace-row"]):
                        # LEFT PANEL: ONLY TWO BIG SCREENS (Upload & Live Preview), NO EXTRA TEXT OR BUTTONS
                        with gr.Column(scale=7, elem_classes=["studio-left-locked"]):
                            with gr.Row(elem_classes=["studio-two-screens-row"]):
                                input_image = gr.Image(
                                    label="📷 Upload Photo",
                                    type="numpy",
                                    image_mode="RGB",
                                    height=440,
                                    elem_id="studio_upload_image",
                                )
                                output_image = gr.Image(**_out_img_kwargs)

                        # RIGHT PANEL: SCROLLABLE CONTROLS
                        with gr.Column(scale=5, elem_classes=["studio-right-scroll"]):
                            # FIRST BUTTON: Run (Manual execution only when clicked)
                            run_btn = gr.Button(
                                "🚀 Run",
                                variant="primary",
                                size="lg",
                            )

                            # DROPDOWN 1: Remove BG & Upscale (Simple colour, 1-10 slider, 3 tick boxes)
                            with gr.Accordion("Remove BG & Upscale", open=True):
                                with gr.Row():
                                    bg_color_dropdown = gr.Dropdown(
                                        choices=bg_preset_choices,
                                        value="#98f3ff",
                                        label="Background Colour",
                                        scale=3,
                                    )
                                    bg_color_picker = gr.ColorPicker(
                                        value="#98f3ff",
                                        label="Colour Picker",
                                        scale=2,
                                        elem_id="mc_bg_picker",
                                    )
                                upscale_slider = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=4,
                                    step=1,
                                    label="AI Upscale & Improvement Quality (1 to 10 — 4 = 2K, 7 = 4K, 10 = True 8K)",
                                )
                                with gr.Row():
                                    auto_level_chk = gr.Checkbox(
                                        value=False,
                                        label="Auto Straighten Head",
                                    )
                                    use_ai_upscale_chk = gr.Checkbox(
                                        value=True,
                                        label="Use AI Upscale Slider",
                                    )
                                    use_full_ai_chk = gr.Checkbox(
                                        value=True,
                                        label="Use Full AI Feature",
                                    )

                            # DROPDOWN 2: Manual Colour Correction (60-FPS Client Live Preview + <85ms PNG Commit)
                            with gr.Accordion("Manual Colour Correction", open=True, elem_id="manual_color_accordion"):
                                with gr.Row():
                                    gamma_slider = gr.Slider(
                                        minimum=0.20,
                                        maximum=2.50,
                                        value=1.00,
                                        step=0.02,
                                        label="Gamma",
                                        elem_id="mc_gamma",
                                    )
                                    bright_slider = gr.Slider(
                                        minimum=-60.0,
                                        maximum=60.0,
                                        value=0.0,
                                        step=1.0,
                                        label="Brightness",
                                        elem_id="mc_bright",
                                    )
                                with gr.Row():
                                    contrast_slider = gr.Slider(
                                        minimum=0.50,
                                        maximum=1.80,
                                        value=1.00,
                                        step=0.02,
                                        label="Contrast",
                                        elem_id="mc_contrast",
                                    )
                                    sat_slider = gr.Slider(
                                        minimum=0.00,
                                        maximum=2.00,
                                        value=1.00,
                                        step=0.02,
                                        label="Saturation",
                                        elem_id="mc_sat",
                                    )
                                with gr.Row():
                                    warmth_slider = gr.Slider(
                                        minimum=-40.0,
                                        maximum=40.0,
                                        value=0.0,
                                        step=1.0,
                                        label="Warmth (Color Temp)",
                                        elem_id="mc_warmth",
                                    )
                                    shadows_slider = gr.Slider(
                                        minimum=-40.0,
                                        maximum=40.0,
                                        value=0.0,
                                        step=1.0,
                                        label="Shadows Lift",
                                        elem_id="mc_shadows",
                                    )
                                reset_color_btn = gr.Button(
                                    "↺ Reset Manual Colour",
                                    variant="secondary",
                                    size="sm",
                                    elem_id="mc_reset_all_btn",
                                )
                                mc_commit_btn = gr.Button(
                                    "Commit Colour",
                                    visible=False,
                                    elem_id="studio_color_commit_btn",
                                )

                            # DROPDOWN 3: Dress Change (Photorealistic AI Try-On + AI Veil/Veli Reminder!)
                            with gr.Accordion("Dress Change", open=False):
                                veil_reminder_md = gr.Markdown("")
                                dress_dropdown = gr.Dropdown(
                                    choices=dress_choices,
                                    value="none",
                                    label="Select Dress (Photorealistic AI Try-On — Suit / Saree / Uniform / Couple)",
                                )
                                dress_target_dd = gr.Dropdown(
                                    choices=dress_target_choices,
                                    value="auto",
                                    label="Target Person (Single or Dual / Couple Photo)",
                                )
                                custom_dress_upload = gr.Image(
                                    label="Optional: Upload Custom Dress Photo (AI Try-On)",
                                    type="numpy",
                                    image_mode="RGB",
                                    height=135,
                                )
                                tryon_dress_btn = gr.Button(
                                    "👗 Try On Dress (AI Preview — No Re-Upscale)",
                                    variant="primary",
                                    size="sm",
                                )
                                with gr.Row():
                                    dress_vshift = gr.Slider(
                                        minimum=-1.0,
                                        maximum=1.0,
                                        value=0.0,
                                        step=0.05,
                                        label="Move Dress Up / Down",
                                    )
                                    dress_sh_scale = gr.Slider(
                                        minimum=0.85,
                                        maximum=1.15,
                                        value=1.00,
                                        step=0.02,
                                        label="Dress Shoulder Width",
                                    )

                            # DROPDOWN 4: Save Customer Record
                            with gr.Accordion("Save Customer", open=False):
                                with gr.Row():
                                    cust_name_input = gr.Textbox(
                                        label="Customer Name",
                                        placeholder="e.g. Anupam Bhowmik",
                                        scale=3,
                                    )
                                    cust_id_state = gr.Textbox(
                                        label="Customer ID (Blank = New)",
                                        placeholder="New or ms-1",
                                        scale=2,
                                    )
                                save_cust_btn = gr.Button("💾 Save Customer", variant="secondary")
                                save_status_md = gr.Markdown("")

                            # DROPDOWN 5: Direct Phone Camera QR Shoot (?mobile_shoot=<session_id>)
                            with gr.Accordion("📱 Phone Camera QR Shoot", open=False, elem_id="phone_qr_accordion"):
                                with gr.Row():
                                    phone_qr_gen_btn = gr.Button(
                                        "📷 Show Phone Shoot QR",
                                        variant="secondary",
                                        elem_id="phone_qr_shoot_btn",
                                    )
                                    phone_load_btn = gr.Button(
                                        "📥 Load Photo from Phone",
                                        variant="primary",
                                    )
                                phone_qr_img = gr.Image(
                                    label="Scan QR on Phone to Shoot Directly",
                                    type="numpy",
                                    height=190,
                                    interactive=False,
                                )
                                phone_session_id = gr.Textbox(
                                    label="Mobile Shoot Session ID",
                                    interactive=True,
                                )
                                phone_status_md = gr.Markdown("")

                            # DROPDOWN 6: Permanent Customer Share Link & 4-Digit PIN Vault (?share=<share_id>)
                            with gr.Accordion("🔗 Share Link & 4-Digit PIN", open=False):
                                with gr.Row():
                                    share_pin_input = gr.Textbox(
                                        label="4-Digit PIN (Blank = Auto)",
                                        placeholder="e.g. 4508",
                                        max_lines=1,
                                        scale=2,
                                    )
                                    create_share_btn = gr.Button(
                                        "🔗 Create Share Link & QR",
                                        variant="primary",
                                        scale=3,
                                    )
                                with gr.Row():
                                    share_url_out = gr.Textbox(
                                        label="Permanent Share Link (?share=...)",
                                        interactive=False,
                                        scale=3,
                                    )
                                    share_pin_out = gr.Textbox(
                                        label="4-Digit PIN",
                                        interactive=False,
                                        scale=1,
                                    )
                                share_qr_img = gr.Image(
                                    label="Customer Share QR Code (Scan to Open & Download)",
                                    type="numpy",
                                    height=190,
                                    interactive=False,
                                )
                                share_status_md = gr.Markdown("")

                # ================================================================
                # TAB 2: SAVED CUSTOMER (SIMPLE, FAST, LIVE SEARCH & 1-CLICK LOAD)
                # ================================================================
                with gr.TabItem("Saved Customer"):
                    with gr.Column(elem_classes=["studio-tab-scrollable"]):
                        with gr.Row():
                            search_box = gr.Textbox(
                                label="🔍 Instant Search by Customer Name or ID (e.g. ms-1)",
                                placeholder="Type name or ID to filter immediately...",
                                scale=4,
                            )
                            load_cid_input = gr.Textbox(
                                label="Selected Customer ID",
                                placeholder="Click any photo or row below",
                                scale=2,
                            )
                            load_cust_btn = gr.Button("📂 Load into Studio", variant="primary", scale=2)
                            del_cust_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        db_action_status = gr.Markdown("")
                        with gr.Row():
                            with gr.Column(scale=6):
                                cust_table = gr.Dataframe(
                                    headers=["ID", "Customer Name", "Saved Date", "Details"],
                                    datatype=["str", "str", "str", "str"],
                                    label="Customer List (Click any row to load)",
                                    interactive=False,
                                )
                            with gr.Column(scale=6):
                                cust_gallery = gr.Gallery(
                                    label="Customer Thumbnails (Click any photo to load)",
                                    columns=4,
                                    height=420,
                                )

                # ================================================================
                # TAB 3: BACKUP
                # ================================================================
                with gr.TabItem("Backup"):
                    with gr.Column(elem_classes=["studio-tab-scrollable"]):
                        with gr.Row():
                            sys_status_json = gr.JSON(label="📊 Cloud & Local Storage Status")
                            with gr.Column():
                                refresh_sys_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                                backup_zip_btn = gr.Button("📦 Create Full ZIP Backup", variant="primary")
                                backup_file_out = gr.File(label="📥 Download ZIP Backup")
                                purge_btn = gr.Button("⚠️ Clear Local Storage", variant="stop")
                                purge_status_md = gr.Markdown("")

                # ================================================================
                # TAB 4: FACE GUIDE
                # ================================================================
                with gr.TabItem("Face Guide"):
                    with gr.Column(elem_classes=["studio-tab-scrollable"]):
                        with gr.Row():
                            telemetry_img = gr.Image(
                                label="📐 Biometric Multi-Subject Eye-Line, Shoulder & Crop Guide",
                                type="numpy",
                                height=440,
                                interactive=False,
                            )
                            telemetry_json = gr.JSON(
                                label="📊 AI Photometric, Multi-Subject & Resolution Telemetry",
                            )

        # ====================================================================
        # EVENT WIRING (MANUAL 'Run', DECOUPLED 'Try On', & <20ms LIVE COLOR)
        # ====================================================================
        if hasattr(input_image, "upload"):
            input_image.upload(fn=lambda: None, inputs=[], outputs=[studio_session_state])
        if hasattr(input_image, "clear"):
            input_image.clear(fn=lambda: (None, None), inputs=[], outputs=[studio_session_state, output_image])

        run_inputs = [
            input_image,
            bg_color_picker,
            upscale_slider,
            auto_level_chk,
            use_ai_upscale_chk,
            use_full_ai_chk,
            gamma_slider,
            bright_slider,
            contrast_slider,
            sat_slider,
            warmth_slider,
            shadows_slider,
            dress_dropdown,
            dress_vshift,
            dress_sh_scale,
            dress_target_dd,
            custom_dress_upload,
            studio_session_state,
        ]
        run_outputs = [
            output_image,
            telemetry_img,
            telemetry_json,
            studio_session_state,
            veil_reminder_md,
        ]

        # 0. Fast (<12ms CPU, Zero-GPU, No Auto-Run!) upload sync + AI Veil (`Veli`) check
        input_image.change(
            fn=_on_upload_sync_and_veil_check,
            inputs=[input_image, dress_dropdown, dress_target_dd, studio_session_state],
            outputs=[studio_session_state, veil_reminder_md],
            show_progress="hidden",
        )

        # 1. Full Studio Creation + Upscaling runs ONLY when user clicks 'Run'
        run_btn.click(
            fn=_ui_run_studio_gradio,
            inputs=run_inputs,
            outputs=run_outputs,
            api_name="_ui_run_studio",
        )

        # 2. Decoupled AI Dress Virtual Try-On (runs on dropdown selection OR button click — NO re-upscaling!)
        tryon_inputs = [
            input_image,
            dress_dropdown,
            dress_target_dd,
            custom_dress_upload,
            dress_vshift,
            dress_sh_scale,
            bg_color_picker,
            gamma_slider,
            bright_slider,
            contrast_slider,
            sat_slider,
            warmth_slider,
            shadows_slider,
            studio_session_state,
        ]
        tryon_outputs = [
            output_image,
            telemetry_img,
            telemetry_json,
            studio_session_state,
            veil_reminder_md,
        ]
        tryon_dress_btn.click(
            fn=_ui_tryon_dress_gradio,
            inputs=tryon_inputs,
            outputs=tryon_outputs,
            api_name="_ui_tryon_dress",
        )
        dress_dropdown.change(
            fn=_ui_tryon_dress_gradio,
            inputs=tryon_inputs,
            outputs=tryon_outputs,
        )

        # 3. Zero-Lag Manual Colour Correction (60-FPS Browser Canvas during drag + single <85ms PNG commit on release!)
        manual_color_fast_inputs = [
            gamma_slider,
            bright_slider,
            contrast_slider,
            sat_slider,
            warmth_slider,
            shadows_slider,
            bg_color_picker,
            studio_session_state,
        ]
        _chg_params = inspect.signature(gamma_slider.change).parameters
        _fast_chg_kw = {"trigger_mode": "always_last"} if "trigger_mode" in _chg_params else {}
        if "show_progress" in _chg_params:
            _fast_chg_kw["show_progress"] = "minimal"

        bg_color_dropdown.change(
            fn=_on_bg_preset_change_gradio,
            inputs=[
                bg_color_dropdown,
                gamma_slider,
                bright_slider,
                contrast_slider,
                sat_slider,
                warmth_slider,
                shadows_slider,
                studio_session_state,
            ],
            outputs=[bg_color_picker, output_image],
        )
        bg_color_picker.change(
            fn=_ui_live_manual_color_gradio,
            inputs=manual_color_fast_inputs,
            outputs=[output_image],
            api_name="_ui_live_manual_color",
            **_fast_chg_kw,
        )
        mc_commit_btn.click(
            fn=_ui_live_manual_color_gradio,
            inputs=manual_color_fast_inputs,
            outputs=[output_image],
            **_fast_chg_kw,
        )
        for mc_slider in (
            gamma_slider,
            bright_slider,
            contrast_slider,
            sat_slider,
            warmth_slider,
            shadows_slider,
        ):
            if hasattr(mc_slider, "release"):
                _rel_params = inspect.signature(mc_slider.release).parameters
                _rel_kw = {}
                if "trigger_mode" in _rel_params:
                    _rel_kw["trigger_mode"] = "always_last"
                if "show_progress" in _rel_params:
                    _rel_kw["show_progress"] = "minimal"
                mc_slider.release(
                    fn=_ui_live_manual_color_gradio,
                    inputs=manual_color_fast_inputs,
                    outputs=[output_image],
                    **_rel_kw,
                )
            else:
                mc_slider.change(
                    fn=_ui_live_manual_color_gradio,
                    inputs=manual_color_fast_inputs,
                    outputs=[output_image],
                    **_fast_chg_kw,
                )

        reset_color_btn.click(
            fn=_ui_reset_manual_color_gradio,
            inputs=[bg_color_picker, studio_session_state],
            outputs=[
                gamma_slider,
                bright_slider,
                contrast_slider,
                sat_slider,
                warmth_slider,
                shadows_slider,
                output_image,
            ],
            api_name="_ui_reset_manual_color",
        )

        # 4. Save Customer & Auto-Refresh Saved Customer List
        save_cust_btn.click(
            fn=_ui_save_customer_v4,
            inputs=[
                cust_name_input,
                input_image,
                bg_color_picker,
                upscale_slider,
                bright_slider,
                contrast_slider,
                sat_slider,
                gamma_slider,
                dress_dropdown,
                cust_id_state,
                warmth_slider,
                shadows_slider,
                auto_level_chk,
                use_ai_upscale_chk,
                use_full_ai_chk,
                dress_vshift,
                dress_sh_scale,
                dress_target_dd,
            ],
            outputs=[save_status_md, cust_id_state, cust_gallery, cust_table],
        )

        # 4b. Direct Phone QR Camera Shoot & Permanent 4-Digit PIN Share Vault wiring
        phone_qr_gen_btn.click(
            fn=_ui_open_mobile_qr_shoot,
            inputs=[],
            outputs=[phone_session_id, phone_qr_img, phone_status_md],
        )
        phone_load_btn.click(
            fn=_ui_load_mobile_shoot_photo,
            inputs=[phone_session_id],
            outputs=[input_image, phone_status_md],
        )
        if hasattr(gr, "Timer"):
            mobile_poll_timer = gr.Timer(value=1.8, active=True)
            mobile_poll_timer.tick(
                fn=_ui_auto_poll_mobile_shoot,
                inputs=[phone_session_id],
                outputs=[input_image, phone_status_md],
            )
        mobile_send_btn.click(
            fn=_ui_submit_mobile_camera_photo,
            inputs=[mobile_shoot_sid_box, mobile_camera_input],
            outputs=[mobile_send_status_md],
        )
        back_to_studio_from_mobile_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            inputs=[],
            outputs=[main_studio_tabs_group, mobile_shoot_portal_group, share_vault_portal_group],
        )
        create_share_btn.click(
            fn=_ui_create_customer_share,
            inputs=[output_image, cust_name_input, share_pin_input, studio_session_state],
            outputs=[share_url_out, share_pin_out, share_qr_img, share_status_md],
        )
        vault_unlock_btn.click(
            fn=_ui_unlock_customer_share,
            inputs=[vault_share_id_box, vault_pin_box],
            outputs=[vault_status_md, vault_deep_zoom_html, vault_download_file],
        )
        back_to_studio_from_vault_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            inputs=[],
            outputs=[main_studio_tabs_group, mobile_shoot_portal_group, share_vault_portal_group],
        )

        # 5. Saved Customer: Instant Live Search + 1-Click Load from Table or Gallery
        search_box.change(
            fn=handle_search_customers,
            inputs=[search_box],
            outputs=[cust_gallery, cust_table],
        )
        search_box.submit(
            fn=handle_search_customers,
            inputs=[search_box],
            outputs=[cust_gallery, cust_table],
        )

        load_outputs = [
            input_image,
            studio_session_state,
            cust_id_state,
            cust_name_input,
            bg_color_dropdown,
            bg_color_picker,
            upscale_slider,
            auto_level_chk,
            use_ai_upscale_chk,
            use_full_ai_chk,
            gamma_slider,
            bright_slider,
            contrast_slider,
            sat_slider,
            warmth_slider,
            shadows_slider,
            dress_dropdown,
            dress_target_dd,
            custom_dress_upload,
            dress_vshift,
            dress_sh_scale,
            db_action_status,
        ]

        load_cust_btn.click(
            fn=_ui_load_customer_v4,
            inputs=[load_cid_input],
            outputs=load_outputs,
        ).then(fn=_ui_run_studio_gradio, inputs=run_inputs, outputs=run_outputs)

        cust_table.select(
            fn=_on_table_select,
            inputs=[cust_table],
            outputs=[load_cid_input],
        ).then(
            fn=_ui_load_customer_v4,
            inputs=[load_cid_input],
            outputs=load_outputs,
        ).then(fn=_ui_run_studio_gradio, inputs=run_inputs, outputs=run_outputs)

        cust_gallery.select(
            fn=_on_gallery_select,
            inputs=[],
            outputs=[load_cid_input],
        ).then(
            fn=_ui_load_customer_v4,
            inputs=[load_cid_input],
            outputs=load_outputs,
        ).then(fn=_ui_run_studio_gradio, inputs=run_inputs, outputs=run_outputs)

        del_cust_btn.click(
            fn=handle_delete_customer,
            inputs=[load_cid_input],
            outputs=[db_action_status, cust_gallery, cust_table],
        )

        # 6. Backup & System Status
        refresh_sys_btn.click(
            fn=handle_refresh_system_status,
            inputs=[],
            outputs=[sys_status_json],
        )
        backup_zip_btn.click(
            fn=create_backup_file_on_disk,
            inputs=[],
            outputs=[backup_file_out],
        )
        purge_btn.click(
            fn=purge_system,
            inputs=[],
            outputs=[purge_status_md],
        )

        demo.load(
            fn=_ui_route_on_load,
            inputs=[],
            outputs=[
                main_studio_tabs_group,
                mobile_shoot_portal_group,
                share_vault_portal_group,
                mobile_shoot_sid_box,
                mobile_shoot_header_md,
                vault_share_id_box,
                vault_header_md,
                phone_session_id,
                phone_qr_img,
                phone_status_md,
            ],
        )
        demo.load(
            fn=handle_search_customers,
            inputs=[search_box],
            outputs=[cust_gallery, cust_table],
        )
        demo.load(
            fn=handle_refresh_system_status,
            inputs=[],
            outputs=[sys_status_json],
        )

    return demo


def _prefetch_cloud_weights_to_disk_cleanly() -> None:
    """
    Gradio 6.32 + ZeroGPU (`spaces` Zygote `os.fork()`) Fork-Safe Disk Prefetcher:
    Downloads model files to `CLOUD_MODELS_DIR` on Hugging Face's fast internal mirror WITHOUT
    spawning background `torch` threads across `demo.launch()` (`os.fork()`) and WITHOUT holding
    5.5 GB of duplicate CPU tensors in the Gradio web-server process!
    """
    if not IS_HF_CLOUD:
        return
    import gc
    from src.shared.config import CLOUD_MODELS_DIR, HF_TOKEN

    os.makedirs(CLOUD_MODELS_DIR, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

        for repo_id, allow in (
            ("1038lab/RMBG-2.0", ["*.json", "*.py", "model.safetensors"]),
            ("mattmdjaga/segformer_b2_clothes", ["*.json", "model.safetensors"]),
            ("stabilityai/sd-vae-ft-mse", ["*.json", "*safetensors*"]),
            ("booksforcharlie/stable-diffusion-inpainting", ["scheduler/*.json", "unet/*.json", "unet/*model*"]),
        ):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=CLOUD_MODELS_DIR,
                    allow_patterns=allow,
                    token=HF_TOKEN,
                )
            except Exception as inner_e:
                print(f"ℹ️ [Prefetch] {repo_id} deferred to on-demand worker: {inner_e}")

        catvton_dir = os.path.join(CLOUD_MODELS_DIR, "catvton_ckpt")
        hf_hub_download(
            repo_id="zhengchong/CatVTON",
            filename="mix-48k-1024/attention/model.safetensors",
            local_dir=catvton_dir,
            token=HF_TOKEN,
        )
        print("✅ [Prefetch] Cloud neural checkpoints cached cleanly on NVMe disk (0 MB web-server RAM).")
    except Exception as e:
        print(f"ℹ️ [Prefetch] Non-fatal prefetch note: {e}")
    finally:
        gc.collect()


demo = build_gradio_app() if HAS_GRADIO else None

if __name__ == "__main__":
    import gc

    print("🚀 Starting Maa Sitala Passport Photo Studio V10 (Gradio 6.32 + Pure-BRIA + CatVTON + 8K Master)...")
    setup_codeformer(auto_install=False)
    pull_from_cloud()
    if IS_HF_CLOUD:
        _prefetch_cloud_weights_to_disk_cleanly()
    gc.collect()
    if demo is None:
        demo = build_gradio_app()
    _launch_sig = inspect.signature(demo.launch).parameters
    _launch_kwargs: dict[str, object] = {
        "server_name": "0.0.0.0",
        "server_port": int(os.environ.get("PORT", 7860)),
    }
    if "ssr_mode" in _launch_sig:
        _launch_kwargs["ssr_mode"] = False
    if "css" in _launch_sig:
        _launch_kwargs["css"] = STUDIO_CSS
    if "head" in _launch_sig:
        _launch_kwargs["head"] = STUDIO_HEAD_HTML
    demo.launch(**_launch_kwargs)
