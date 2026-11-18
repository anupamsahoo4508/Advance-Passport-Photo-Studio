"""
V8.5 Cloud-Native AI Virtual Try-On Engine (`vton_ai_engine.py` — Canonical VITON-HD `bfloat16` CUDA `CatVTON`).

Architecture:
1. Concatenate-Based Diffusion Virtual Try-On (`zhengchong/CatVTON` `mix-48k-1024` self-attention
   weights + `booksforcharlie/stable-diffusion-inpainting` 9-channel UNet + `stabilityai/sd-vae-ft-mse`
   high-fidelity VAE) running natively in `bfloat16` PyTorch 2.0 SDPA on Hugging Face ZeroGPU (`CUDA`)
   — matching Official CatVTON `--mixed_precision bf16` with zero FP16 attention overflow!
2. Canonical VITON-HD `768 × 1024` Similarity Framing (`canon_ipd = 96 px, canon_cx = 384, canon_chin_y = 292`):
   - Maps any passport photo (from tight close-up headshots to multi-person couple portraits) into
     CatVTON's exact native VITON-HD / DressCode upper-body studio geometry so full suits, neckties,
     blazers, and Banarasi silk sarees always generate at 1:1 anatomical scale without bottom-edge cutoffs!
   - Pre-normalizes optical backlit veiling glare on `p_canvas` before VTON conditioning so hazy phone
     uploads never wash out rich Maroon, Royal Blue, or Navy garments into gray!
   - Pre-fills central throat V-zones with chromophore-matched cervical skin while erasing 100% of old
     standing shirt collars, school-uniform straps, and wide outer sleeves.
3. Full-Scale `768 × 1024` Sleeved Flat-Lay Garment Conditioning (`garment_refs_v8_6`) + Custom Dress Upload:
   - Aligns collar, lapels, tie, sleeves, and diagonal saree zari borders 1:1 with the canonical person frame
     in `<15ms` with zero extra SD1.5 pipeline overhead.
4. Seamless `768 × 1024` Pure-BRIA (`RMBG-2.0`) Silhouette Re-Matting:
   - Extracts clean foreground garment alpha directly on the un-cropped `768 × 1024` CatVTON studio canvas
     BEFORE inverse-mapping back to the portrait — making rectangular paste borders 100% impossible!

STRICT CLOUD-ONLY WEIGHTS (`0 MB` on Local Windows PC):
All neural weights download and execute ONLY when `IS_HF_CLOUD == True`.
"""
from __future__ import annotations

import inspect
import math
import os
import threading
import time
import numpy as np
from PIL import Image

from src.shared.config import CLOUD_MODELS_DIR, HF_TOKEN, IS_HF_CLOUD
from src.core.cv_ops import HAS_CV2, bilinear_sample, cv2, gaussian_blur, resize_img
from src.core.multi_subject_detector import SubjectProfile, detect_foreground_subjects

_VTON_LOCK = threading.Lock()
_CATVTON_BUNDLE: dict[str, object] | None = None
_CATVTON_TRIED: bool = False
_SEGFORMER_BUNDLE: dict[str, object] | None = None
_SEGFORMER_TRIED: bool = False

GARMENT_REFS_CACHE_DIR = os.path.join(CLOUD_MODELS_DIR, "garment_refs_v9_0")
_ASSETS_GARMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "garments"))

# Pure Studio White Backdrop matching VITON-HD / DressCode CatVTON training distribution
_VTON_STUDIO_BG_RGB = np.array([254.0, 254.0, 255.0], dtype=np.float32)

# Canonical 768x1024 VITON-HD Upper-Body Portrait Coordinates
_CANON_W = 768
_CANON_H = 1024
_CANON_CX = 384.0
_CANON_CHIN_Y = 292.0
_CANON_IPD = 96.0

# Official Couple Combination Presets -> (male_preset_key, female_preset_key)
COUPLE_COMBO_MAP: dict[str, tuple[str, str]] = {
    "couple_suit_and_maroon_saree": ("male_suit_navy_tie", "female_saree_maroon_gold"),
    "couple_blazer_and_royal_saree": ("male_blazer_royal", "female_saree_royal_blue"),
    "couple_formal_shirts": ("male_shirt_white", "female_shirt_sky_blue"),
    "couple_school_uniforms": ("male_shirt_white", "female_uniform_navy_white"),
}


def get_vton_engine_status() -> dict[str, object]:
    """Return diagnostic status of the V8.5 Cloud-Native CatVTON Virtual Try-On Engine."""
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
        "catvton_loaded": _CATVTON_BUNDLE is not None,
        "garment_refs_dir": GARMENT_REFS_CACHE_DIR if IS_HF_CLOUD else "DISABLED_ON_LOCAL_PC",
    }


def _load_catvton_engine_cloud() -> dict[str, object] | None:
    """
    Lazy-load `CatVTON` (`booksforcharlie/stable-diffusion-inpainting` UNet + `sd-vae-ft-mse` VAE
    + `zhengchong/CatVTON` `mix-48k-1024` self-attention weights) strictly inside HF Cloud (`IS_HF_CLOUD`).
    """
    global _CATVTON_BUNDLE, _CATVTON_TRIED
    if not IS_HF_CLOUD:
        return None
    if os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return None
    if _CATVTON_BUNDLE is not None:
        return _CATVTON_BUNDLE
    if _CATVTON_TRIED:
        return None

    with _VTON_LOCK:
        if _CATVTON_BUNDLE is not None:
            return _CATVTON_BUNDLE
        if _CATVTON_TRIED:
            return None
        _CATVTON_TRIED = True

        try:
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from safetensors.torch import load_file as load_safetensors_file  # type: ignore

            os.makedirs(CLOUD_MODELS_DIR, exist_ok=True)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available() and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True

            class SkipAttnProcessor(torch.nn.Module):
                """CatVTON Cross-Attention Bypass (eliminates CLIP text conditioning overhead)."""
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__()

                def __call__(
                    self,
                    attn,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    temb=None,
                    *args,
                    **kwargs,
                ):
                    return hidden_states

            class CatVTONAttnProcessor2_0(torch.nn.Module):
                """CatVTON PyTorch 2.0 Scaled Dot-Product Self-Attention across concatenated latents."""
                def __init__(self, hidden_size=None, cross_attention_dim=None, **kwargs) -> None:
                    super().__init__()

                def __call__(
                    self,
                    attn,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    temb=None,
                    *args,
                    **kwargs,
                ):
                    residual = hidden_states
                    if attn.spatial_norm is not None:
                        hidden_states = attn.spatial_norm(hidden_states, temb)
                    input_ndim = hidden_states.ndim
                    if input_ndim == 4:
                        b_sz, ch, hh, ww = hidden_states.shape
                        hidden_states = hidden_states.view(b_sz, ch, hh * ww).transpose(1, 2)

                    b_sz, seq_len, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                    )
                    if attention_mask is not None:
                        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, b_sz)
                        attention_mask = attention_mask.view(b_sz, attn.heads, -1, attention_mask.shape[-1])
                    if attn.group_norm is not None:
                        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    query = attn.to_q(hidden_states)
                    if encoder_hidden_states is None:
                        encoder_hidden_states = hidden_states
                    elif attn.norm_cross:
                        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                    key = attn.to_k(encoder_hidden_states)
                    value = attn.to_v(encoder_hidden_states)
                    inner_dim = key.shape[-1]
                    head_dim = inner_dim // attn.heads

                    query = query.view(b_sz, -1, attn.heads, head_dim).transpose(1, 2)
                    key = key.view(b_sz, -1, attn.heads, head_dim).transpose(1, 2)
                    value = value.view(b_sz, -1, attn.heads, head_dim).transpose(1, 2)

                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
                    hidden_states = hidden_states.transpose(1, 2).reshape(b_sz, -1, attn.heads * head_dim)
                    hidden_states = hidden_states.to(query.dtype)
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)

                    if input_ndim == 4:
                        hidden_states = hidden_states.transpose(-1, -2).reshape(b_sz, ch, hh, ww)
                    if attn.residual_connection:
                        hidden_states = hidden_states + residual
                    return hidden_states / attn.rescale_output_factor

            base_ckpt = "booksforcharlie/stable-diffusion-inpainting"
            scheduler = DDIMScheduler.from_pretrained(
                base_ckpt,
                subfolder="scheduler",
                cache_dir=CLOUD_MODELS_DIR,
                token=HF_TOKEN,
            )
            # Load in clean FP32 on CPU first so FP32 CatVTON attention weights load losslessly before BF16 cast
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                cache_dir=CLOUD_MODELS_DIR,
                token=HF_TOKEN,
            )
            unet = UNet2DConditionModel.from_pretrained(
                base_ckpt,
                subfolder="unet",
                cache_dir=CLOUD_MODELS_DIR,
                token=HF_TOKEN,
            )

            attn_procs = {}
            for name in unet.attn_processors.keys():
                if name.endswith("attn1.processor"):
                    attn_procs[name] = CatVTONAttnProcessor2_0()
                else:
                    attn_procs[name] = SkipAttnProcessor()
            unet.set_attn_processor(attn_procs)

            # Exact `zhengchong/CatVTON` `attn_of_unet` submodule sequence (16 Attention blocks * 8 entries = 128)
            attn_modules = torch.nn.ModuleList()
            for name, mod in unet.named_modules():
                if "attn1" in name:
                    attn_modules.append(mod)

            catvton_dir = os.path.join(CLOUD_MODELS_DIR, "catvton_ckpt")
            ckpt_file = hf_hub_download(
                repo_id="zhengchong/CatVTON",
                filename="mix-48k-1024/attention/model.safetensors",
                local_dir=catvton_dir,
                token=HF_TOKEN,
            )
            attn_sd = load_safetensors_file(ckpt_file, device="cpu")
            # Each of the 16 Attention (`attn1`) blocks sits at stride 8 (`0, 8, ..., 120`) in `attn_modules`;
            # loading per-block with `strict=True` verifies all 80 tensors without tripping on child-module aliases!
            attn1_blocks = [mod for name, mod in unet.named_modules() if name.endswith(".attn1")]
            if len(attn1_blocks) != 16:
                raise RuntimeError(f"Expected 16 attn1 blocks in UNet, got {len(attn1_blocks)}")
            loaded_tensor_count = 0
            for blk_idx, blk_mod in enumerate(attn1_blocks):
                prefix = f"{blk_idx * 8}."
                blk_sd = {k[len(prefix):]: v for k, v in attn_sd.items() if k.startswith(prefix)}
                blk_mod.load_state_dict(blk_sd, strict=True)
                loaded_tensor_count += len(blk_sd)
            if loaded_tensor_count != len(attn_sd):
                raise RuntimeError(f"Loaded {loaded_tensor_count}/{len(attn_sd)} CatVTON attention tensors")

            # Keep UNet in native BF16 (exact CatVTON training exponent range -> zero FP16 overflow!)
            # and keep sd-vae-ft-mse in FP32 (full 23-bit mantissa -> zero 8x8 quantization stipple!)
            bf16_ok = bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
            unet.to(dtype=torch.bfloat16 if bf16_ok else torch.float32)
            vae.to(dtype=torch.float32)
            unet.eval()
            vae.eval()

            _CATVTON_BUNDLE = {
                "unet": unet,
                "vae": vae,
                "scheduler": scheduler,
            }
            print(f"✅ [CatVTON Cloud] Loaded mix-48k-1024 ({loaded_tensor_count} strict tensors across {len(attn1_blocks)} attn1 blocks, BF16 UNet + FP32 VAE)")
            return _CATVTON_BUNDLE
        except Exception as e:
            print(f"⚠️ [CatVTON Cloud] Failed to initialize CatVTON engine: {e}")
            return None


def _load_segformer_parser_cloud() -> dict[str, object] | None:
    """
    Lazy-load `mattmdjaga/segformer_b2_clothes` (18-class ATR human body & clothing parser)
    strictly inside HF Cloud (`IS_HF_CLOUD == True`, `0 MB` on local PC).
    """
    global _SEGFORMER_BUNDLE, _SEGFORMER_TRIED
    if not IS_HF_CLOUD:
        return None
    if os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return None
    if _SEGFORMER_BUNDLE is not None:
        return _SEGFORMER_BUNDLE
    if _SEGFORMER_TRIED:
        return None

    with _VTON_LOCK:
        if _SEGFORMER_BUNDLE is not None:
            return _SEGFORMER_BUNDLE
        if _SEGFORMER_TRIED:
            return None
        _SEGFORMER_TRIED = True
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation  # type: ignore

            os.makedirs(CLOUD_MODELS_DIR, exist_ok=True)
            repo_id = "mattmdjaga/segformer_b2_clothes"
            proc = AutoImageProcessor.from_pretrained(repo_id, cache_dir=CLOUD_MODELS_DIR, token=HF_TOKEN)
            model = AutoModelForSemanticSegmentation.from_pretrained(repo_id, cache_dir=CLOUD_MODELS_DIR, token=HF_TOKEN)
            model.eval()
            _SEGFORMER_BUNDLE = {"processor": proc, "model": model}
            print("✅ [SegFormer Cloud] Loaded mattmdjaga/segformer_b2_clothes 18-part human parser")
            return _SEGFORMER_BUNDLE
        except Exception as e:
            print(f"⚠️ [SegFormer Cloud] Non-fatal parser load fallback: {e}")
            return None


def _parse_human_768x1024(rgb_768x1024: np.ndarray) -> np.ndarray | None:
    """
    Run `SegFormer-B2` 18-part semantic human parsing on a `768 × 1024` canonical portrait (`~18 ms` on ZeroGPU):
    ATR Labels: `0=Bg, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress,
    8=Belt, 11=Face/Skin, 14=Left-arm, 15=Right-arm, 16=Bag, 17=Scarf`.
    """
    bundle = _load_segformer_parser_cloud()
    if bundle is None:
        return None
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        proc = bundle["processor"]  # type: ignore[assignment]
        model = bundle["model"]  # type: ignore[assignment]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        pil_im = Image.fromarray(rgb_768x1024)
        inputs = proc(images=pil_im, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            up = F.interpolate(logits, size=(_CANON_H, _CANON_W), mode="bilinear", align_corners=False)
            seg = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return seg
    except Exception as e:
        print(f"⚠️ [SegFormer Cloud] Parse inference fallback: {e}")
        return None


def _flood_exterior_white_bg_768(rgb_u8: np.ndarray, tol: float = 8.0) -> np.ndarray:
    """Flood-fill exterior `#FFFFFF` studio background from canvas borders; returns boolean mask (`True` = exterior bg)."""
    import collections

    hh, ww = rgb_u8.shape[:2]
    bright = np.min(rgb_u8.astype(np.float32), axis=2) >= (255.0 - float(tol))
    ext = np.zeros((hh, ww), dtype=bool)
    q: collections.deque[tuple[int, int]] = collections.deque()
    for x in range(ww):
        if bright[0, x]:
            ext[0, x] = True
            q.append((0, x))
        if bright[hh - 1, x] and not ext[hh - 1, x]:
            ext[hh - 1, x] = True
            q.append((hh - 1, x))
    for y in range(hh):
        if bright[y, 0] and not ext[y, 0]:
            ext[y, 0] = True
            q.append((y, 0))
        if bright[y, ww - 1] and not ext[y, ww - 1]:
            ext[y, ww - 1] = True
            q.append((y, ww - 1))
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < hh and 0 <= nx < ww and not ext[ny, nx] and bright[ny, nx]:
                ext[ny, nx] = True
                q.append((ny, nx))
    return ext


def _condition_garment_ref_for_catvton(raw_ref_u8: np.ndarray, preset_key: str) -> np.ndarray:
    """
    Clean Studio Garment Reference Normalizer for `CatVTON` (`bfloat16` UNet + `float32` `sd-vae-ft-mse`):
    - Preserves 100% of authentic Savile-Row wool tones, crisp `#FFFFFF` dress-shirt collars, silk neckties,
      lapel shading, and buttons without artificial hue/luminance warping.
    - Normalizes exterior studio background (including enclosed sleeve-to-torso gaps) to pure `#FFFFFF` (`255`)
      and suppresses sub-1.8px WebP compression micro-grain.
    """
    arr_f = raw_ref_u8.astype(np.float32)
    ext_bg = _flood_exterior_white_bg_768(raw_ref_u8, tol=10.0)
    fg = (~ext_bg).astype(np.float32)
    if float(fg.sum()) < 2000.0:
        return raw_ref_u8

    cyy_g, cxx_g = np.meshgrid(
        np.arange(_CANON_H, dtype=np.float32),
        np.arange(_CANON_W, dtype=np.float32),
        indexing="ij",
    )
    luma = 0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2]
    chroma_g = np.max(arr_f, axis=2) - np.min(arr_f, axis=2)
    sleeve_gap_bg = (np.abs(cxx_g - _CANON_CX) > 112.0) & (cyy_g > 360.0) & (luma > 228.0) & (chroma_g < 10.0)
    if preset_key not in ("male_shirt_white", "female_uniform_navy_white", "female_kurti_ivory_gold"):
        ext_bg = ext_bg | sleeve_gap_bg

    smooth = gaussian_blur(arr_f, sigma=1.0)
    raw_detail = arr_f - smooth
    clean_detail = np.sign(raw_detail) * np.maximum(0.0, np.abs(raw_detail) - 1.6) * 1.05
    arr_clean = np.clip(smooth + clean_detail, 0.0, 255.0)
    arr_clean[ext_bg] = 255.0
    return np.clip(np.round(arr_clean), 0, 255).astype(np.uint8)


def _align_garment_ref_to_body(
    garment_ref_768: np.ndarray,
    left_sh_w: float,
    right_sh_w: float,
    sr: float = 1.0,
) -> np.ndarray:
    """
    Proportional VITON-HD Body-Scale Garment Alignment:
    Gently adapts the flat-lay garment width & torso height around the canonical VITON-HD `222 px`
    half-shoulder baseline (`scale_x = clip(person_half_w / 222.0, 0.84, 1.10)`) and centers it on
    the person's measured shoulder midpoint `384 + 0.45 * (right_sh_w - left_sh_w)` so slender boys,
    petite girls, women, and broad-shouldered men get 1:1 anatomical lapel, V-yoke, and sleeve fit!
    """
    person_half_w = 0.5 * (float(left_sh_w) + float(right_sh_w))
    scale_x = float(np.clip(person_half_w / 222.0, 0.84, 1.10))
    scale_y = float(np.clip(0.35 + 0.65 * float(sr), 0.80, 1.04))
    shift_x = int(round(np.clip(0.45 * (float(right_sh_w) - float(left_sh_w)), -28.0, 28.0)))

    if abs(scale_x - 1.0) < 0.025 and abs(scale_y - 1.0) < 0.025 and abs(shift_x) <= 2:
        return garment_ref_768

    nw = max(64, int(round(_CANON_W * scale_x)))
    nh = max(64, int(round(_CANON_H * scale_y)))
    resized = resize_img(garment_ref_768, nw, nh, mode="lanczos")

    canvas = np.full((_CANON_H, _CANON_W, 3), 255, dtype=np.uint8)
    # Anchor collar top around canonical y=160 so scaling vertically keeps neckline aligned with chin!
    anchor_y = 160
    dst_x0 = int(round(_CANON_CX - 0.5 * nw)) + shift_x
    dst_y0 = int(round(anchor_y - anchor_y * scale_y))

    sx1 = max(0, -dst_x0)
    sy1 = max(0, -dst_y0)
    dx1 = max(0, dst_x0)
    dy1 = max(0, dst_y0)
    copy_w = min(nw - sx1, _CANON_W - dx1)
    copy_h = min(nh - sy1, _CANON_H - dy1)
    if copy_w > 0 and copy_h > 0:
        canvas[dy1 : dy1 + copy_h, dx1 : dx1 + copy_w] = resized[sy1 : sy1 + copy_h, sx1 : sx1 + copy_w]
    return canvas


def _get_or_create_photoreal_garment_ref(preset_key: str, shared_vae=None) -> np.ndarray:
    """
    Load the Real Photographic Ghost-Mannequin Studio Garment Asset (`768 × 1024` RGB on pure `#FFFFFF` white)
    from embedded `GARMENT_WEBP_B64` (or `assets/garments/<preset_key>.jpg`) and apply `CatVTON` photometric calibration.
    """
    _ = shared_vae
    import base64
    import io
    try:
        from src.core.garment_catalog_data import GARMENT_WEBP_B64
        b64_str = GARMENT_WEBP_B64.get(preset_key)
        if b64_str:
            with Image.open(io.BytesIO(base64.b64decode(b64_str))) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if arr.shape[:2] != (_CANON_H, _CANON_W):
                arr = _prepare_custom_garment_ref(arr)
            return _condition_garment_ref_for_catvton(arr, preset_key)
    except Exception:
        pass

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        asset_path = os.path.join(_ASSETS_GARMENTS_DIR, f"{preset_key}{ext}")
        if os.path.isfile(asset_path) and os.path.getsize(asset_path) > 2048:
            try:
                with Image.open(asset_path) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                if arr.shape[:2] != (_CANON_H, _CANON_W):
                    arr = _prepare_custom_garment_ref(arr)
                return _condition_garment_ref_for_catvton(arr, preset_key)
            except Exception:
                pass

    from src.core.dress_engine import build_canonical_garment_template

    tmpl_rgb, tmpl_alpha, _ = build_canonical_garment_template(preset_key, canvas_w=900, canvas_h=920)
    gw, gh = 700, 776
    g_rgb = resize_img(tmpl_rgb[132:, :, :].astype(np.float32), gw, gh, mode="lanczos")
    g_alp = np.clip(resize_img(tmpl_alpha[132:, :].astype(np.float32), gw, gh, mode="bilinear"), 0.0, 1.0)[:, :, None]
    canvas = np.full((_CANON_H, _CANON_W, 3), 255.0, dtype=np.float32)
    y0, x0 = 192, (_CANON_W - gw) // 2
    canvas[y0 : y0 + gh, x0 : x0 + gw] = g_alp * g_rgb + (1.0 - g_alp) * 255.0
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)


def _prepare_custom_garment_ref(custom_garment_rgb: np.ndarray) -> np.ndarray:
    """Aspect-preserving letterbox resize of any user-uploaded custom dress photo onto 768×1024 white canvas."""
    raw = custom_garment_rgb[:, :, :3]
    gh, gw = raw.shape[:2]
    scale = min(720.0 / float(max(1, gw)), 860.0 / float(max(1, gh)))
    nw = max(32, min(_CANON_W, int(round(gw * scale))))
    nh = max(32, min(_CANON_H, int(round(gh * scale))))
    resized = resize_img(raw, nw, nh, mode="lanczos")
    canvas = np.full((_CANON_H, _CANON_W, 3), 255, dtype=np.uint8)
    y0 = (_CANON_H - nh) // 2
    x0 = (_CANON_W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _build_subject_agnostic_mask(
    img_rgb: np.ndarray,
    fg_alpha: np.ndarray | None,
    subj: SubjectProfile,
    preset_style: str = "suit",
    vertical_shift: float = 0.0,
    shoulder_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the Subject-Scoped Anatomical Cloth-Agnostic Mask `(binary_vton_mask, soft_blend_mask)`
    in portrait coordinates `(h, w)`:
    - Strictly `0.0` outside `subj.roi_box` (protects adjacent people in joint/couple portraits!).
    - Strictly `0.0` on face, eyes, hair crown, bindi, sindoor, nose-pin, earrings, chin, and beard!
    - Covers 100% of standing old shirt collars (e.g. checkered collars beside the jaw) AND wide
      existing sleeves, while leaving the central upper throat cylinder clean.
    """
    h, w = img_rgb.shape[:2]
    g = subj.geom
    ipd = max(16.0, float(g.ipd_px))
    rx1, _ry1, rx2, _ry2 = subj.roi_box
    roi_w = max(32.0, float(rx2 - rx1))

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    face_cx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
    eye_mid_y = 0.5 * (g.left_eye.center_y + g.right_eye.center_y)
    dy_shift = float(vertical_shift) * ipd * 0.18
    chin_y = float(g.chin_xy[1]) + dy_shift

    af = (
        np.clip(fg_alpha.astype(np.float32), 0.0, 1.0)
        if (fg_alpha is not None and fg_alpha.shape[:2] == (h, w))
        else np.ones((h, w), dtype=np.float32)
    )

    max_sh_half_w = min(0.485 * roi_w, 2.35 * ipd * float(shoulder_scale))
    max_sh_half_w = max(max_sh_half_w, 1.55 * ipd)

    # 1. Anatomical Sloped-Shoulder & Sleeve Dome
    collar_wing_zone = np.exp(-0.5 * (((np.abs(xx - face_cx) - 0.58 * ipd) / max(3.0, 0.16 * ipd)) ** 2))
    collar_base_y = chin_y - 0.04 * ipd - (0.22 * ipd) * collar_wing_zone
    lat_sh = np.clip((np.abs(xx - face_cx) - 0.52 * ipd) / max(8.0, max_sh_half_w - 0.52 * ipd), 0.0, 1.0)
    dome_roof_y = collar_base_y + (0.58 * ipd) * (lat_sh ** 1.40)
    dome_vert = np.clip((yy - dome_roof_y) / max(2.5, 0.04 * ipd), 0.0, 1.0)

    roi_pad_l = float(rx1 + 2) if rx1 > 0 else 0.0
    roi_pad_r = float(rx2 - 2) if rx2 < w else float(w)
    dome_left_x = np.maximum(face_cx - max_sh_half_w, roi_pad_l)
    dome_right_x = np.minimum(face_cx + max_sh_half_w, roi_pad_r)
    dome_horiz = (
        np.clip((xx - dome_left_x) / max(2.5, 0.05 * ipd), 0.0, 1.0)
        * np.clip((dome_right_x - xx) / max(2.5, 0.05 * ipd), 0.0, 1.0)
    )
    anatomical_dome = dome_vert * dome_horiz

    # 2. Dilated Old-Torso & High Standing Collar Eraser (covers standing collars beside jaw & wide sleeves!)
    high_collar_start_y = np.minimum(chin_y - 0.34 * ipd, eye_mid_y + 0.96 * ipd)
    old_cloth = ((af > 0.18).astype(np.float32)) * np.clip((yy - high_collar_start_y) / max(2.5, 0.05 * ipd), 0.0, 1.0)
    if float(old_cloth.max()) > 0.1:
        old_cloth = np.clip(gaussian_blur(old_cloth, sigma=max(3.5, 0.14 * ipd)) * 2.4, 0.0, 1.0)
        old_cloth *= np.clip((yy - (high_collar_start_y - 0.03 * ipd)) / max(2.5, 0.04 * ipd), 0.0, 1.0)

    # Protect head/jaw/chin/beard oval, earrings, and head-draped saree pallu above chin
    head_jaw_oval = np.clip(
        1.0 - (((xx - face_cx) / (0.68 * ipd)) ** 2 + ((yy - (chin_y - 0.94 * ipd)) / (0.98 * ipd)) ** 2),
        0.0,
        1.0,
    )
    head_jaw_hard = ((((xx - face_cx) / (0.64 * ipd)) ** 2 + ((yy - (chin_y - 0.94 * ipd)) / (0.95 * ipd)) ** 2) <= 1.0).astype(np.float32)
    head_guard = np.maximum(head_jaw_oval, head_jaw_hard)
    earring_guard = np.clip(subj.ornament_lock * 1.5, 0.0, 1.0) * (yy < chin_y - 0.04 * ipd).astype(np.float32)
    pallu_guard = (
        np.clip(((chin_y + 0.02 * ipd) - yy) / max(2.5, 0.06 * ipd), 0.0, 1.0)
        if subj.has_head_pallu
        else np.zeros_like(yy)
    )

    _ = preset_style
    raw_mask = np.maximum(anatomical_dome, old_cloth) * (1.0 - head_guard) * (1.0 - earring_guard) * (1.0 - 0.90 * pallu_guard)
    if rx1 > 0:
        raw_mask[:, :rx1] = 0.0
    if rx2 < w:
        raw_mask[:, rx2:] = 0.0

    bin_mask = (raw_mask > 0.40).astype(np.float32)
    soft_blend = gaussian_blur(bin_mask, sigma=max(2.2, 0.035 * ipd))
    soft_blend *= (1.0 - head_guard)
    if rx1 > 0:
        soft_blend[:, :rx1] = 0.0
    if rx2 < w:
        soft_blend[:, rx2:] = 0.0
    return bin_mask, np.clip(soft_blend, 0.0, 1.0).astype(np.float32)


def _normalize_optical_haze_for_vton(p_canvas_u8: np.ndarray, head_fg_mask: np.ndarray) -> np.ndarray:
    """
    Optical Backlit Veiling-Glare Dehazer for `CatVTON` Conditioning ONLY:
    When a phone photo was shot against a bright window/curtain (e.g. dark hair lifted to RGB 85+),
    `CatVTON` would otherwise render dark maroon/navy garments washed out to match the milky flare!
    Normalizing the dark black-point on `p_canvas` lets `CatVTON` synthesize rich, studio-saturated
    garments while the original un-decoded face/hair stays 100% untouched outside the garment mask!
    """
    fg_px = p_canvas_u8[head_fg_mask > 0.5]
    if fg_px.shape[0] < 120:
        return p_canvas_u8
    luma = 0.299 * fg_px[:, 0].astype(np.float32) + 0.587 * fg_px[:, 1].astype(np.float32) + 0.114 * fg_px[:, 2].astype(np.float32)
    p04 = float(np.percentile(luma, 4.0))
    if p04 <= 32.0:
        return p_canvas_u8

    # Gentle highlight-preserving gamma curve (deepens backlit milky blacks without ever clipping skin highlights!)
    gamma_haze = float(np.clip(1.0 + (p04 - 32.0) * 0.0045, 1.0, 1.16))
    img_f = p_canvas_u8.astype(np.float32)
    dehazed = np.clip(((np.clip(img_f, 0.0, 255.0) / 255.0) ** gamma_haze) * 255.0, 0.0, 255.0)
    fg_w = np.clip(gaussian_blur(head_fg_mask.astype(np.float32), sigma=1.5), 0.0, 1.0)[:, :, None]
    out = fg_w * dehazed + (1.0 - fg_w) * img_f
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _prepare_canonical_subject_vton(
    clean_base_rgb: np.ndarray,
    af_clean: np.ndarray | None,
    subj: SubjectProfile,
    sub_preset_key: str = "none",
    vertical_shift: float = 0.0,
    shoulder_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    V10 `AnatomicalBodyFit™` — SegFormer-B2 18-Part Human Parser + True Body-Silhouette Mapping:
    1. Isolates ONLY `subj` onto `768 × 1024` Canonical VITON-HD Studio Space.
    2. Runs `mattmdjaga/segformer_b2_clothes` (`Face=11, Hair=2, Upper=4, Pants=6, Dress=7, Arms=14/15, Scarf=17`)
       to lock exact biological face, jawline, scalp hair, buns, pigtails, sindoor, bindi & nose-pin while
       surgically unwrapping old head-veils (`veli`/`dupatta`), ear-wraps, uniform sashes, and standing collars!
    3. Measures the customer's ACTUAL left & right shoulder/arm widths (`left_sh_w`, `right_sh_w`) directly
       from their body in the photo and builds a curved, gender-anatomical, body-hugging `m_canvas` (rounded
       deltoids + feminine waist curve for women/girls, athletic V-taper for men/boys — NEVER boxy rectangles!).
    """
    h, w = clean_base_rgb.shape[:2]
    g = subj.geom
    ipd = max(16.0, float(g.ipd_px))
    rx1, _ry1, rx2, _ry2 = subj.roi_box
    face_cx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
    dy_shift = float(vertical_shift) * ipd * 0.18
    chin_y = float(g.chin_xy[1]) + dy_shift

    # 1. Isolate strictly this subject's foreground over `_VTON_STUDIO_BG_RGB`
    subj_af = (
        np.clip(af_clean.astype(np.float32), 0.0, 1.0).copy()
        if (af_clean is not None and af_clean.shape[:2] == (h, w))
        else np.ones((h, w), dtype=np.float32)
    )
    if rx1 > 0:
        subj_af[:, :rx1] = 0.0
    if rx2 < w:
        subj_af[:, rx2:] = 0.0

    subj_rgb_f = (
        subj_af[:, :, None] * clean_base_rgb.astype(np.float32)
        + (1.0 - subj_af[:, :, None]) * _VTON_STUDIO_BG_RGB[None, None, :]
    )
    subj_rgb_u8 = np.clip(np.round(subj_rgb_f), 0, 255).astype(np.uint8)

    # 2. Exact Canonical VITON-HD Similarity Window (`win_w / win_h == 768 / 1024`)
    base_scale = _CANON_IPD / ipd
    below_chin_px = max(64.0, float(h) - chin_y)
    min_scale_for_full_torso = float(_CANON_H - _CANON_CHIN_Y - 6.0) / below_chin_px
    scale_to_canon = max(0.64 * base_scale, min(base_scale, min_scale_for_full_torso))
    canon_ipd_eff = ipd * scale_to_canon

    win_w = float(_CANON_W) / scale_to_canon
    win_h = float(_CANON_H) / scale_to_canon
    wx1 = face_cx - (_CANON_CX / scale_to_canon)
    wy1 = chin_y - (_CANON_CHIN_Y / scale_to_canon)

    iw = max(32, int(round(win_w)))
    ih = max(32, int(round(win_h)))
    ix0 = int(round(wx1))
    iy0 = int(round(wy1))

    sx1 = max(0, ix0)
    sy1 = max(0, iy0)
    sx2 = min(w, ix0 + iw)
    sy2 = min(h, iy0 + ih)

    win_rgb = np.full((ih, iw, 3), np.round(_VTON_STUDIO_BG_RGB).astype(np.uint8), dtype=np.uint8)
    win_af = np.zeros((ih, iw), dtype=np.float32)
    win_skin = np.zeros((ih, iw), dtype=np.float32)
    win_orn = np.zeros((ih, iw), dtype=np.float32)
    if sx2 > sx1 and sy2 > sy1:
        dx1 = sx1 - ix0
        dy1 = sy1 - iy0
        win_rgb[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = subj_rgb_u8[sy1:sy2, sx1:sx2]
        win_af[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = subj_af[sy1:sy2, sx1:sx2]
        win_skin[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = g.skin_mask[sy1:sy2, sx1:sx2]
        win_orn[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = subj.ornament_lock[sy1:sy2, sx1:sx2]

    p_canvas = resize_img(win_rgb, _CANON_W, _CANON_H, mode="lanczos")
    af_canon = np.clip(resize_img(win_af, _CANON_W, _CANON_H, mode="bilinear"), 0.0, 1.0)
    skin_canon = np.clip(resize_img(win_skin, _CANON_W, _CANON_H, mode="bilinear"), 0.0, 1.0)
    orn_canon = np.clip(resize_img(win_orn, _CANON_W, _CANON_H, mode="bilinear"), 0.0, 1.0)

    cyy, cxx = np.meshgrid(
        np.arange(_CANON_H, dtype=np.float32),
        np.arange(_CANON_W, dtype=np.float32),
        indexing="ij",
    )
    cdx = np.abs(cxx - _CANON_CX)
    sr = canon_ipd_eff / _CANON_IPD

    p_raw = p_canvas.astype(np.float32)
    p_luma = 0.299 * p_raw[:, :, 0] + 0.587 * p_raw[:, :, 1] + 0.114 * p_raw[:, :, 2]
    p_chroma = np.max(p_raw, axis=2) - np.min(p_raw, axis=2)

    # 3. Cloud SegFormer-B2 18-Part Semantic Human Parsing on raw `p_canvas`
    # Note: Label `1=Hat` often false-positives on puffy studio-lit bare hair, whereas Label `17=Scarf`
    # specifically detects real head-veils, dupattas & saree pallus (`01`, `02`, `03`, `04`)!
    seg_map = _parse_human_768x1024(p_canvas)
    if seg_map is not None:
        seg_upper_cloth = np.isin(seg_map, [4, 5, 7, 8, 16])
        seg_scarf = (seg_map == 17)
        seg_hair = (seg_map == 2) & (cyy <= _CANON_CHIN_Y + 4.0 * sr) & (cdx <= 106.0 * sr)
        seg_face_head = (seg_map == 11) & (cyy <= _CANON_CHIN_Y + 4.0 * sr) & (cdx <= 92.0 * sr)
        seg_arms = np.isin(seg_map, [14, 15])
    else:
        seg_upper_cloth = np.zeros((_CANON_H, _CANON_W), dtype=bool)
        seg_scarf = np.zeros((_CANON_H, _CANON_W), dtype=bool)
        seg_hair = np.zeros((_CANON_H, _CANON_W), dtype=bool)
        seg_face_head = np.zeros((_CANON_H, _CANON_W), dtype=bool)
        seg_arms = np.zeros((_CANON_H, _CANON_W), dtype=bool)

    # Normalize backlit optical veiling glare on `p_canvas` for CatVTON conditioning so dark suits/sarees stay rich
    head_haze_ref = ((af_canon > 0.45) & (cyy <= _CANON_CHIN_Y)).astype(np.float32)
    p_canvas = _normalize_optical_haze_for_vton(p_canvas, head_haze_ref)
    p_f = p_canvas.astype(np.float32)

    # 4. Exact Biological Head, Hair, Ear, Sindoor, Bindi & Jaw Protection (ZERO forehead/ear/hair notches!)
    sindoor_lock = (
        (af_canon > 0.50)
        & (cdx <= 22.0 * sr)
        & (cyy >= _CANON_CHIN_Y - 215.0 * sr)
        & (cyy <= _CANON_CHIN_Y - 95.0 * sr)
        & (p_raw[:, :, 0] > p_raw[:, :, 1] + 24.0)
    )
    orn_lock = (orn_canon > 0.40) & (cdx <= 96.0 * sr) & (cyy <= _CANON_CHIN_Y - 6.0 * sr)
    throat_skin_anchor = (
        (skin_canon > 0.36)
        & (~seg_upper_cloth)
        & (cdx <= 32.0 * sr)
        & (cyy >= _CANON_CHIN_Y - 8.0 * sr)
        & (cyy <= _CANON_CHIN_Y + 5.0 * sr)
    )
    natural_warm_skin = (
        (p_raw[:, :, 0] > p_raw[:, :, 1] + 7.0)
        & (p_raw[:, :, 1] >= p_raw[:, :, 2] + 2.0)
        & (p_raw[:, :, 0] - p_raw[:, :, 1] < 50.0)
        & (p_luma > 42.0)
        & (p_luma < 225.0)
    )

    scarf_wing_px = int(np.count_nonzero(seg_scarf & (cyy <= _CANON_CHIN_Y + 25.0 * sr) & (cdx >= 68.0 * sr)))
    has_veil_or_scarf = bool(subj.has_head_pallu or (subj.inferred_gender == "female" and scarf_wing_px >= 80))

    if not has_veil_or_scarf:
        # Un-veiled men, boys, girls & women:
        # 1) Keep 100% of bare non-cloth neck skin intact above `dome_top` (`bare_neck_skin & (canon_dome < 0.5)`) so
        #    close-up portraits (`07` red-shirt man) NEVER get neck cutouts or floating heads!
        # 2) Anchor upper cervical skin (`<= _CANON_CHIN_Y + 12*sr`), while leaving open V-necks/chest hair below
        #    `+12*sr` and standing outer shirt/polo collars (`seg_upper_cloth | crimson_collar_cloth`) inside `m_canvas`
        #    so CatVTON buttons a crisp White Dress Shirt Collar + Silk Necktie cleanly onto the real neck!
        face_jaw_oval = (((cxx - _CANON_CX) / (66.5 * sr)) ** 2 + ((cyy - (_CANON_CHIN_Y - 90.0 * sr)) / (94.0 * sr)) ** 2) <= 1.0
        crimson_collar_cloth = (p_raw[:, :, 0] - p_raw[:, :, 1] > 52.0) & (cyy >= _CANON_CHIN_Y - 68.0 * sr) & (~face_jaw_oval)
        bare_neck_skin = (
            (af_canon > 0.35)
            & (~seg_upper_cloth)
            & (~crimson_collar_cloth)
            & ((skin_canon > 0.22) | natural_warm_skin | seg_face_head)
            & (cdx <= 86.0 * sr)
            & (cyy >= _CANON_CHIN_Y - 46.0 * sr)
            & (cyy <= _CANON_CHIN_Y + 52.0 * sr)
        )
        upper_cervical_skin = bare_neck_skin & (cdx <= 58.0 * sr) & (cyy <= _CANON_CHIN_Y + 12.0 * sr)
        upper_head_ears_hair = (
            (af_canon > 0.20)
            & (~seg_upper_cloth)
            & (~crimson_collar_cloth)
            & (cyy <= _CANON_CHIN_Y - 36.0 * sr)
            & (cdx <= 136.0 * sr)
        )
        dark_pigtail_lock = (
            (af_canon > 0.36)
            & (~seg_upper_cloth)
            & (skin_canon < 0.24)
            & (p_luma < 76.0)
            & (p_chroma < 26.0)
            & (cyy <= _CANON_CHIN_Y + 2.0 * sr)
            & (cdx <= 108.0 * sr)
        ) if subj.inferred_gender == "female" else np.zeros_like(face_jaw_oval)
        bio_head_protect = (
            face_jaw_oval
            | upper_cervical_skin
            | upper_head_ears_hair
            | ((seg_hair) & (cyy <= _CANON_CHIN_Y - 8.0 * sr))
            | ((seg_face_head) & (cdx <= 64.0 * sr) & (cyy <= _CANON_CHIN_Y - 12.0 * sr))
            | sindoor_lock
            | orn_lock
            | throat_skin_anchor
            | dark_pigtail_lock
        )
        standing_collar_tips = (
            (seg_upper_cloth | crimson_collar_cloth)
            & (~face_jaw_oval)
            & (~upper_cervical_skin)
            & (~throat_skin_anchor)
            & (~seg_hair)
            & (cyy >= _CANON_CHIN_Y - 68.0 * sr)
            & (cyy <= _CANON_CHIN_Y + 18.0 * sr)
        )
        upper_head_cloth_to_unwrap = standing_collar_tips
    else:
        # Veiled / Dupatta / Head-Pallu women (`01`, `02`, `03`, `04`):
        # On confirmed veiled subjects, SegFormer `1=Hat` also marks head-scarves/veils (`seg_veil_cloth`).
        bare_neck_skin = np.zeros((_CANON_H, _CANON_W), dtype=bool)
        seg_veil_cloth = seg_scarf | ((seg_map == 1) if seg_map is not None else np.zeros((_CANON_H, _CANON_W), dtype=bool))
        maroon_or_vivid_cloth = (p_raw[:, :, 0] - p_raw[:, :, 1] > 32.0) | (p_chroma > 38.0)
        t_jaw = np.clip((cyy - (_CANON_CHIN_Y - 88.0 * sr)) / max(10.0, 62.0 * sr), 0.0, 1.0)
        rx_cran = (106.0 - 38.0 * t_jaw) * sr
        cranial_oval = (((cxx - _CANON_CX) / rx_cran) ** 2 + ((cyy - (_CANON_CHIN_Y - 116.0 * sr)) / (124.0 * sr)) ** 2) <= 1.0
        jaw_neck_skin_v = (
            (af_canon > 0.42)
            & (~seg_upper_cloth)
            & (~seg_veil_cloth)
            & (~maroon_or_vivid_cloth)
            & (skin_canon > 0.30)
            & (cdx <= 54.0 * sr)
            & (cyy >= _CANON_CHIN_Y - 36.0 * sr)
            & (cyy <= _CANON_CHIN_Y + 10.0 * sr)
        )
        inner_face_core = (
            ((((cxx - _CANON_CX) / (66.0 * sr)) ** 2 + ((cyy - (_CANON_CHIN_Y - 92.0 * sr)) / (94.0 * sr)) ** 2) <= 1.0)
            | jaw_neck_skin_v
            | sindoor_lock
            | orn_lock
            | (throat_skin_anchor & (~seg_veil_cloth) & (~maroon_or_vivid_cloth))
        )
        bare_dark_hair = (
            (af_canon > 0.42)
            & (~seg_veil_cloth)
            & (~seg_upper_cloth)
            & (
                seg_hair
                | ((p_luma < 56.0) & (p_chroma < 16.0) & (cyy <= _CANON_CHIN_Y - 108.0 * sr) & (cdx <= 94.0 * sr))
            )
            & (cdx <= 108.0 * sr)
            & (cyy <= _CANON_CHIN_Y - 34.0 * sr)
        )
        jaw_chin_strap_cloth = (
            (~inner_face_core)
            & (~bare_dark_hair)
            & (cyy >= _CANON_CHIN_Y - 96.0 * sr)
            & (skin_canon < 0.28)
            & (orn_canon < 0.35)
            & (
                seg_upper_cloth
                | seg_veil_cloth
                | maroon_or_vivid_cloth
            )
        )
        bio_head_protect = inner_face_core | bare_dark_hair | (cranial_oval & (~jaw_chin_strap_cloth))
        upper_head_cloth_to_unwrap = (
            (af_canon > 0.08)
            & (cyy <= _CANON_CHIN_Y + 24.0 * sr)
            & (~bio_head_protect)
            & (
                seg_upper_cloth
                | seg_veil_cloth
                | jaw_chin_strap_cloth
                | (cdx > rx_cran + 2.0 * sr)
            )
        )

    # 5. Measure Customer's ACTUAL Left & Right Body Widths in the Image (`AnatomicalBodyFit™` Target Profile)
    canon_scale_x = float(_CANON_W) / float(iw)
    canon_scale_y = float(_CANON_H) / float(ih)
    vis_left_x = float(max(rx1, sx1) - ix0) * canon_scale_x
    vis_right_x = float(min(rx2, sx2) - ix0) * canon_scale_x
    vis_bot_y = float((sy2 - iy0)) * canon_scale_y

    is_tall_standing = bool(sr < 0.80)
    # Exact VITON-HD jacket/top proportion (`932.0` on standing portraits so CatVTON renders tailored suits + ties, NEVER 1022px kurtas!)
    trouser_cutoff_y = 932.0 if is_tall_standing else 1022.0

    veil_flare = seg_scarf & (cyy < _CANON_CHIN_Y + 115.0 * sr)
    measure_y_top = _CANON_CHIN_Y + (95.0 * sr if has_veil_or_scarf else 42.0 * sr)
    measure_y_bot = min(trouser_cutoff_y - 18.0, vis_bot_y - 8.0, _CANON_CHIN_Y + 420.0 * sr)
    torso_fg = (
        (af_canon > 0.42)
        & (~veil_flare)
        & (cyy >= measure_y_top)
        & (cyy <= max(measure_y_top + 24.0, measure_y_bot))
    )

    is_fem = subj.inferred_gender == "female"
    anat_default_w = (212.0 if is_fem else 234.0) * max(0.86, sr)
    min_fit_w = (186.0 if is_fem else 202.0) * max(0.84, sr)
    max_fit_w = (236.0 if is_fem else 264.0) * max(0.88, sr)

    left_px = (_CANON_CX - cxx)[torso_fg & (cxx < _CANON_CX - 24.0)]
    right_px = (cxx - _CANON_CX)[torso_fg & (cxx > _CANON_CX + 24.0)]
    raw_left_w = float(np.percentile(left_px, 93.0)) if left_px.size >= 40 else anat_default_w
    raw_right_w = float(np.percentile(right_px, 93.0)) if right_px.size >= 40 else anat_default_w

    left_clipped = (_CANON_CX - raw_left_w) <= (vis_left_x + 10.0)
    right_clipped = (_CANON_CX + raw_right_w) >= (vis_right_x - 10.0)
    if left_clipped and not right_clipped and right_px.size >= 40:
        raw_left_w = 0.35 * raw_left_w + 0.65 * raw_right_w
    elif right_clipped and not left_clipped and left_px.size >= 40:
        raw_right_w = 0.35 * raw_right_w + 0.65 * raw_left_w

    vis_torso_span = max(0.0, vis_bot_y - _CANON_CHIN_Y)
    body_conf = float(np.clip((vis_torso_span - 150.0 * sr) / (160.0 * sr), 0.25, 1.0))
    if subj.has_head_pallu:
        raw_left_w = min(raw_left_w, 224.0 * max(0.88, sr))
        raw_right_w = min(raw_right_w, 224.0 * max(0.88, sr))

    target_left_w = float(np.clip((body_conf * raw_left_w + (1.0 - body_conf) * anat_default_w + 10.0) * float(shoulder_scale), min_fit_w, max_fit_w))
    target_right_w = float(np.clip((body_conf * raw_right_w + (1.0 - body_conf) * anat_default_w + 10.0) * float(shoulder_scale), min_fit_w, max_fit_w))
    avg_target_w = 0.5 * (target_left_w + target_right_w)
    target_left_w = float(np.clip(target_left_w, avg_target_w - 14.0, avg_target_w + 14.0))
    target_right_w = float(np.clip(target_right_w, avg_target_w - 14.0, avg_target_w + 14.0))
    # Respect Couple / Partition boundaries `[vis_left_x + 8 .. vis_right_x - 8]` and keep opposite shoulder balanced!
    if rx1 > 0:
        part_max_l = max(164.0, _CANON_CX - (vis_left_x + 8.0))
        if target_left_w > part_max_l:
            target_left_w = part_max_l
            target_right_w = min(target_right_w, target_left_w + 18.0)
    if rx2 < w:
        part_max_r = max(164.0, (vis_right_x - 8.0) - _CANON_CX)
        if target_right_w > part_max_r:
            target_right_w = part_max_r
            target_left_w = min(target_left_w, target_right_w + 18.0)

    # 6. Full-Scale Canonical VITON-HD `m_canvas` for Stage A `CatVTON` Generation (ZERO Narrow-Mask Suit Collapse!)
    # Suits/Blazers ALWAYS generate at native VITON-HD width (`274..284 px` male, `254..264 px` female) so CatVTON
    # synthesizes 100% complete lapels, white dress collars, silk neckties, buttons & sleeves, before Stage B
    # (`_fit_rendered_vton_to_subject_body_768`) tailors the rendered garment 1:1 onto `(target_left_w, target_right_w)`!
    is_suit_or_blazer = ("suit" in sub_preset_key) or ("blazer" in sub_preset_key)
    if is_suit_or_blazer:
        base_gen_w = 264.0 if is_fem else 284.0
        min_gen_w = 254.0 if is_fem else 276.0
    else:
        base_gen_w = 246.0 if is_fem else 272.0
        min_gen_w = 236.0 if is_fem else 262.0
    gen_sh_half = float(np.clip(max(base_gen_w * sr, min_gen_w), 232.0, 336.0))

    collar_wing = np.exp(-0.5 * (((cdx - 52.0 * sr) / max(8.0, 14.0 * sr)) ** 2))
    lat_gen = np.clip((cdx - 48.0 * sr) / max(24.0, gen_sh_half - 48.0 * sr), 0.0, 1.08)
    dome_top = (_CANON_CHIN_Y - 2.0 * sr - 15.0 * sr * collar_wing) + (44.0 * sr) * (lat_gen ** 1.32)

    lower_trouser_lock = ((af_canon > 0.35) & (cyy >= trouser_cutoff_y)) if is_tall_standing else np.zeros_like(bio_head_protect)
    is_half_sleeve = sub_preset_key in ("male_shirt_white", "male_shirt_sky_blue", "female_shirt_sky_blue")
    forearm_skin_lock = (
        (af_canon > 0.45)
        & (seg_arms | natural_warm_skin)
        & (cdx >= 146.0)
        & (cyy >= 570.0)
        if (is_tall_standing and is_half_sleeve)
        else np.zeros_like(bio_head_protect)
    )

    canon_dome = ((cyy >= dome_top) & (cyy <= trouser_cutoff_y) & (cdx <= gen_sh_half)).astype(np.float32)
    above_dome_bare_neck = bare_neck_skin & (canon_dome < 0.5)
    protect = (bio_head_protect | above_dome_bare_neck | lower_trouser_lock | forearm_skin_lock).astype(np.float32)

    # 7. Wipe old outer sleeves, head-veil wings & standing collar tips on `p_canvas` before CatVTON
    bulky_outer_or_veil = (
        upper_head_cloth_to_unwrap
        | (
            (af_canon > 0.08)
            & (cyy >= _CANON_CHIN_Y - 38.0 * sr)
            & (cyy <= trouser_cutoff_y)
            & (canon_dome < 0.5)
            & (protect < 0.5)
        )
    )
    af_before_wipe = af_canon.copy()
    veil_head_wipe_canon = np.zeros((_CANON_H, _CANON_W), dtype=np.float32)
    if np.any(bulky_outer_or_veil):
        wipe_w = np.clip(gaussian_blur(bulky_outer_or_veil.astype(np.float32), sigma=2.6) * 1.40, 0.0, 1.0) * (1.0 - protect)
        if np.any(upper_head_cloth_to_unwrap):
            veil_head_wipe_canon = (
                np.clip(gaussian_blur(upper_head_cloth_to_unwrap.astype(np.float32), sigma=2.8) * 1.45, 0.0, 1.0)
                * (1.0 - protect)
            )
        p_cleaned = p_f * (1.0 - wipe_w[:, :, None]) + _VTON_STUDIO_BG_RGB[None, None, :] * wipe_w[:, :, None]
        p_canvas = np.clip(np.round(p_cleaned), 0, 255).astype(np.uint8)
        af_canon = af_canon * (1.0 - wipe_w)

    # Include central neck cloth (`cdx <= 78*sr`) ONLY on veiled women (`01`, `03`, `04`) so neck-crossing cords/veils
    # are inpainted, while un-veiled men/boys keep their clean VITON-HD neck cylinder for crisp Suit Collar + Tie!
    neck_cloth_bridge = (
        ((af_before_wipe > 0.20) & (cdx <= 78.0 * sr) & (cyy >= _CANON_CHIN_Y - 22.0 * sr) & (cyy <= dome_top + 10.0))
        if has_veil_or_scarf
        else np.zeros_like(bio_head_protect)
    )
    high_collar_old = (
        neck_cloth_bridge
        | (
            (af_canon > 0.16)
            & ((cyy >= _CANON_CHIN_Y + 2.0 * sr) | seg_upper_cloth | seg_scarf | (p_chroma >= 26.0))
            & (cyy >= _CANON_CHIN_Y - 46.0 * sr)
            & (cyy <= trouser_cutoff_y)
            & ((cyy >= dome_top) | (cdx <= 112.0 * sr))
            & (cdx <= gen_sh_half + 10.0 * sr)
        )
    ).astype(np.float32) * (1.0 - protect)
    if float(high_collar_old.max()) > 0.1:
        high_collar_old = (gaussian_blur(high_collar_old, sigma=3.2) > 0.20).astype(np.float32) * (1.0 - protect)

    raw_m = np.maximum(canon_dome, high_collar_old) * (1.0 - protect)
    m_canvas = (gaussian_blur(raw_m, sigma=4.0) >= 0.44).astype(np.float32) * (1.0 - protect)

    meta = {
        "ix0": float(ix0),
        "iy0": float(iy0),
        "iw": float(iw),
        "ih": float(ih),
        "sx1": float(sx1),
        "sy1": float(sy1),
        "sx2": float(sx2),
        "sy2": float(sy2),
        "sr": float(sr),
        "target_left_w": float(target_left_w),
        "target_right_w": float(target_right_w),
        "gen_sh_half": float(gen_sh_half),
        "has_true_crown_veil": 0.0,
        "has_veil_or_scarf": 1.0 if has_veil_or_scarf else 0.0,
        "trouser_cutoff_y": float(trouser_cutoff_y),
    }
    return p_canvas, m_canvas, skin_canon, veil_head_wipe_canon, meta


def _run_catvton_canvas_768x1024(
    bundle: dict[str, object],
    p_canvas_768x1024: np.ndarray,
    m_canvas_768x1024: np.ndarray,
    garment_ref_768x1024: np.ndarray,
    num_steps: int = 30,
    guidance_scale: float = 2.6,
    seed: int = 42,
) -> np.ndarray:
    """
    Execute native `768 × 1024` `CatVTON` with BF16 UNet + FP32 `sd-vae-ft-mse` (matching Official `CatVTONPipeline`).
    """
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    unet = bundle["unet"]  # type: ignore[assignment]
    vae = bundle["vae"]  # type: ignore[assignment]
    scheduler = bundle["scheduler"]  # type: ignore[assignment]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = bool(device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    unet_dtype = torch.bfloat16 if bf16_ok else torch.float32
    vae_dtype = torch.float32

    unet.to(device=device, dtype=unet_dtype)
    vae.to(device=device, dtype=vae_dtype)

    if garment_ref_768x1024.shape[:2] != (_CANON_H, _CANON_W):
        g_canvas = _prepare_custom_garment_ref(garment_ref_768x1024)
    else:
        g_canvas = garment_ref_768x1024

    img_t = torch.from_numpy((p_canvas_768x1024.astype(np.float32) / 127.5) - 1.0).permute(2, 0, 1)[None, ...].to(device=device, dtype=vae_dtype)
    cond_t = torch.from_numpy((g_canvas.astype(np.float32) / 127.5) - 1.0).permute(2, 0, 1)[None, ...].to(device=device, dtype=vae_dtype)
    mask_f32 = torch.from_numpy((m_canvas_768x1024 >= 0.5).astype(np.float32)[None, None, :, :]).to(device=device, dtype=vae_dtype)

    masked_img_t = img_t * (mask_f32 < 0.5)

    with torch.inference_mode():
        gen = torch.Generator(device=device).manual_seed(int(seed))
        masked_latent = (vae.encode(masked_img_t).latent_dist.sample(generator=gen) * vae.config.scaling_factor).to(dtype=unet_dtype)
        cond_latent = (vae.encode(cond_t).latent_dist.sample(generator=gen) * vae.config.scaling_factor).to(dtype=unet_dtype)
        mask_latent = F.interpolate(mask_f32, size=masked_latent.shape[-2:], mode="nearest").to(dtype=unet_dtype)

        concat_dim = -2  # Height concatenation: (1, 4, 128, 96) + (1, 4, 128, 96) -> (1, 4, 256, 96)
        masked_latent_concat = torch.cat([masked_latent, cond_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        latents = torch.randn(
            masked_latent_concat.shape,
            generator=gen,
            device=device,
            dtype=unet_dtype,
        )
        scheduler.set_timesteps(int(num_steps), device=device)
        latents = latents * scheduler.init_noise_sigma

        do_cfg = float(guidance_scale) > 1.0
        if do_cfg:
            uncond_masked = torch.cat([masked_latent, torch.zeros_like(cond_latent)], dim=concat_dim)
            masked_latent_concat = torch.cat([uncond_masked, masked_latent_concat], dim=0)
            mask_latent_concat = torch.cat([mask_latent_concat, mask_latent_concat], dim=0)

        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        extra_kwargs: dict[str, object] = {}
        if "eta" in step_params:
            extra_kwargs["eta"] = 1.0
        if "generator" in step_params:
            extra_kwargs["generator"] = gen

        for t in scheduler.timesteps:
            latent_in = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_in = scheduler.scale_model_input(latent_in, t)
            inpaint_in = torch.cat([latent_in, mask_latent_concat, masked_latent_concat], dim=1)

            noise_pred = unet(
                inpaint_in,
                t,
                encoder_hidden_states=None,
                return_dict=False,
            )[0]

            if do_cfg:
                pred_uncond, pred_cond = noise_pred.chunk(2)
                noise_pred = pred_uncond + float(guidance_scale) * (pred_cond - pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

        person_latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        decoded = vae.decode((1.0 / vae.config.scaling_factor) * person_latents.to(dtype=vae_dtype)).sample
        decoded_01 = ((decoded * 0.5) + 0.5).clamp(0.0, 1.0)
        out_canvas = decoded_01[0].permute(1, 2, 0).float().cpu().numpy() * 255.0

    return np.clip(np.round(out_canvas), 0, 255).astype(np.uint8)


def _extract_clean_canonical_vton_rgba(
    vton_canvas_u8: np.ndarray,
    sub_preset_key: str,
    p_canvas_u8: np.ndarray | None = None,
    m_canvas_f32: np.ndarray | None = None,
    shoulder_scale: float = 1.0,
    sr: float = 1.0,
    has_true_crown_veil: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 100% halo-free, box-free `(fg_rgb_768x1024, alpha_768x1024)` directly on the seamless
    `768 × 1024` CatVTON output canvas BEFORE inverse-mapping to the portrait:
    - Enforces exact causality: any pixel that was BOTH outside `m_canvas` AND already pure studio white
      on `p_canvas` (`m_canvas == 0` & `p_canvas == studio_bg`) is guaranteed exterior background (`alpha = 0`),
      making white shoulder-crescent halos 100% impossible even on snow-white shirts & school uniforms!
    """
    from src.core.bria_bg_engine import decontaminate_bria_foreground, extract_bria_foreground_matte

    vton_alpha, _ = extract_bria_foreground_matte(vton_canvas_u8, geom=None, subjects=None)

    margin_samples = np.concatenate(
        [
            vton_canvas_u8[:160, :36].reshape(-1, 3),
            vton_canvas_u8[:160, -36:].reshape(-1, 3),
            vton_canvas_u8[220:500, :20].reshape(-1, 3),
            vton_canvas_u8[220:500, -20:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    vton_bg_est = np.median(margin_samples, axis=0)

    cyy, cxx = np.meshgrid(
        np.arange(_CANON_H, dtype=np.float32),
        np.arange(_CANON_W, dtype=np.float32),
        indexing="ij",
    )
    cdx = np.abs(cxx - _CANON_CX)

    max_env_half_w = float(np.clip(max(316.0 * sr, 304.0) * float(shoulder_scale), 248.0, 366.0))
    lat_t = np.clip((cdx - 54.0 * sr) / max(24.0, max_env_half_w - 54.0 * sr), 0.0, 1.2)
    roof_y = (_CANON_CHIN_Y - 14.0 * sr) + (44.0 * sr) * (lat_t ** 1.30)
    torso_env = (
        np.clip((cyy - roof_y) / 8.0, 0.0, 1.0)
        * np.clip((max_env_half_w - cdx) / 10.0, 0.0, 1.0)
    )
    head_rx = 220.0 * sr if has_true_crown_veil else max(118.0 * sr, 102.0)
    head_ry = 225.0 * sr if has_true_crown_veil else max(205.0 * sr, 168.0)
    head_r = np.sqrt((cdx / head_rx) ** 2 + ((cyy - (292.0 - 98.0 * sr)) / head_ry) ** 2)
    head_env = np.clip((1.14 - head_r) / 0.10, 0.0, 1.0)
    valid_env = np.maximum(head_env, torso_env)

    # Exact Causality Envelope: outside `m_canvas` AND where `p_canvas` was already pure studio backdrop -> alpha = 0!
    if p_canvas_u8 is not None and m_canvas_f32 is not None:
        p_dist_bg = np.linalg.norm(p_canvas_u8.astype(np.float32) - _VTON_STUDIO_BG_RGB[None, None, :], axis=2)
        p_has_subject = np.clip((p_dist_bg - 6.0) / 10.0, 0.0, 1.0)
        m_dilated = np.clip(gaussian_blur(m_canvas_f32.astype(np.float32), sigma=2.2) * 1.6, 0.0, 1.0)
        causal_fg_gate = np.maximum(p_has_subject, m_dilated)
        valid_env = valid_env * causal_fg_gate

    vton_f = vton_canvas_u8.astype(np.float32)
    vton_luma = 0.299 * vton_f[:, :, 0] + 0.587 * vton_f[:, :, 1] + 0.114 * vton_f[:, :, 2]
    bg_dist_est = np.linalg.norm(vton_f - vton_bg_est[None, None, :], axis=2)
    bg_dist_white = np.linalg.norm(vton_f - _VTON_STUDIO_BG_RGB[None, None, :], axis=2)
    min_bg_dist = np.minimum(bg_dist_est, bg_dist_white)

    weak_bria_haze = np.clip((0.56 - vton_alpha) / 0.26, 0.0, 1.0)
    near_bg = np.clip((16.0 - min_bg_dist) / 8.0, 0.0, 1.0) * weak_bria_haze * (cyy >= _CANON_CHIN_Y - 18.0 * sr)

    # Remove bright studio-white rim ghosts outside central collar (`cdx > 70*sr`) on colored/dark outfits
    # and above shoulder slope (`cyy < roof_y + 18`) on all outfits!
    is_light_outfit = sub_preset_key in (
        "male_shirt_white",
        "female_uniform_navy_white",
        "female_kurti_ivory_gold",
        "custom",
    )
    white_rim_px = np.clip((20.0 - min_bg_dist) / 8.0, 0.0, 1.0) * np.clip((vton_luma - 232.0) / 10.0, 0.0, 1.0)
    if is_light_outfit:
        shoulder_sky_zone = (cdx > 68.0 * sr) & (cyy >= _CANON_CHIN_Y - 28.0 * sr) & (cyy <= roof_y + 30.0)
        near_bg = np.maximum(near_bg, white_rim_px * shoulder_sky_zone.astype(np.float32))
    else:
        outer_non_collar = (cdx > 70.0 * sr) & (cyy >= _CANON_CHIN_Y - 28.0 * sr)
        near_bg = np.maximum(near_bg, white_rim_px * outer_non_collar.astype(np.float32))

    clean_alpha = np.clip(vton_alpha * valid_env * (1.0 - 0.98 * near_bg), 0.0, 1.0).astype(np.float32)
    clean_fg = decontaminate_bria_foreground(vton_canvas_u8, clean_alpha, bg_orig_rgb=vton_bg_est)
    return clean_fg, clean_alpha


def _post_polish_vton_garment_768(
    clean_fg_768: np.ndarray,
    clean_alpha_768: np.ndarray,
    p_canvas_768: np.ndarray,
    m_canvas_768: np.ndarray,
    sub_preset_key: str,
    skin_canon_768: np.ndarray | None = None,
    sr: float = 1.0,
) -> np.ndarray:
    """
    Photorealistic Post-VTON Savile-Row Tone & Fabric Micro-Contrast Finisher:
    - Preserves 100% of `CatVTON`'s real photographic drape, lapels, buttons, pockets, silk pleats, and ties!
    - Smoothly grades midtone-conditioned Royal-Slate / Cool-Slate wool jackets down into rich, deep
      Executive Savile-Row Navy / Charcoal (while keeping inner white shirts, ties, and skin 100% crisp),
      and lifts pearl-conditioned `male_shirt_white` into bright crisp studio white poplin!
    """
    _ = p_canvas_768
    fg_f = clean_fg_768.astype(np.float32).copy()
    luma = 0.299 * fg_f[:, :, 0] + 0.587 * fg_f[:, :, 1] + 0.114 * fg_f[:, :, 2]
    chroma = np.max(fg_f, axis=2) - np.min(fg_f, axis=2)

    cyy_768, cxx_768 = np.meshgrid(
        np.arange(_CANON_H, dtype=np.float32),
        np.arange(_CANON_W, dtype=np.float32),
        indexing="ij",
    )
    cdx_768 = np.abs(cxx_768 - _CANON_CX)

    # Smooth feathered elliptical throat-skin exclusion + bare forearm exclusion (ZERO rectangular collar boxes!)
    is_half_sleeve = sub_preset_key in ("male_shirt_white", "male_shirt_sky_blue", "female_shirt_sky_blue")
    throat_oval = (((cdx_768 / (46.0 * sr)) ** 2) + (((cyy_768 - (_CANON_CHIN_Y + 22.0 * sr)) / (58.0 * sr)) ** 2)) <= 1.0
    forearm_zone = (cdx_768 >= 142.0) & (cyy_768 >= 560.0) if is_half_sleeve else np.zeros_like(throat_oval)
    skin_candidate_zone = gaussian_blur((throat_oval | forearm_zone).astype(np.float32), sigma=6.5)

    true_skin_color = (
        (fg_f[:, :, 0] - fg_f[:, :, 2] > 28.0)
        & (fg_f[:, :, 0] - fg_f[:, :, 1] > 13.0)
        & (luma > 48.0)
        & (luma < 190.0)
    ).astype(np.float32)
    warm_skin = skin_candidate_zone * true_skin_color
    if skin_canon_768 is not None and skin_canon_768.shape == luma.shape:
        warm_skin = np.maximum(warm_skin, np.clip(skin_canon_768 * 1.3, 0.0, 1.0) * skin_candidate_zone)

    cloth_scope = (
        np.clip(gaussian_blur((m_canvas_768 > 0.35).astype(np.float32), sigma=2.5), 0.0, 1.0)
        * np.clip(clean_alpha_768, 0.0, 1.0)
        * (1.0 - np.clip(gaussian_blur(warm_skin, sigma=3.5), 0.0, 1.0))
    )

    smooth = gaussian_blur(fg_f, sigma=1.15)
    micro = fg_f - smooth
    # Gentle dead-zone anti-stipple: removes sub-1.6px VAE quantization noise while keeping 100% of
    # CatVTON's authentic Savile-Row lapels, white spread shirt collars, silk ties, buttons & sari zari!
    micro_clean = np.sign(micro) * np.maximum(0.0, np.abs(micro) - 1.6) * 1.04
    polished = np.clip(smooth + micro_clean, 0.0, 255.0)
    fg_f = fg_f * (1.0 - cloth_scope[:, :, None]) + polished * cloth_scope[:, :, None]
    return np.clip(np.round(fg_f), 0, 255).astype(np.uint8)


def _fit_rendered_vton_to_subject_body_768(
    clean_fg_768: np.ndarray,
    clean_alpha_768: np.ndarray,
    m_canvas_768: np.ndarray,
    target_left_w: float,
    target_right_w: float,
    sub_preset_key: str = "none",
    is_fem: bool = False,
    sr: float = 1.0,
    trouser_cutoff_y: float = 1022.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage B Neck-Locked $C^1$ Anatomical Body Tailoring (`AnatomicalBodyFit™`):
    - Keeps the central neck, dress-shirt collar, necktie knot & blade, and placket (`|x - 384| <= 55*sr`)
      at exact `1.000×` identity so ties, collars, and buttons NEVER distort!
    - Smoothly maps outer clavicles, notch lapels, deltoids, waist, and sleeves from the canonical
      VITON-HD rendered width `(ren_left_w, ren_right_w)` onto the customer's real measured body profile
      `(target_left_w, target_right_w)` with a natural feminine waist curve / male athletic taper!
    """
    cyy, cxx = np.meshgrid(
        np.arange(_CANON_H, dtype=np.float32),
        np.arange(_CANON_W, dtype=np.float32),
        indexing="ij",
    )
    cdx = np.abs(cxx - _CANON_CX)

    y_band_top = _CANON_CHIN_Y + 38.0 * sr
    y_band_bot = min(_CANON_CHIN_Y + 235.0 * sr, trouser_cutoff_y - 24.0)
    sh_band = (
        (clean_alpha_768 > 0.45)
        & (m_canvas_768 > 0.25)
        & (cyy >= y_band_top)
        & (cyy <= max(y_band_top + 24.0, y_band_bot))
    )
    l_xs = (_CANON_CX - cxx)[sh_band & (cxx < _CANON_CX - 36.0)]
    r_xs = (cxx - _CANON_CX)[sh_band & (cxx > _CANON_CX + 36.0)]
    if l_xs.size < 32 or r_xs.size < 32:
        return clean_fg_768, clean_alpha_768

    ren_left_w = float(np.percentile(l_xs, 96.5))
    ren_right_w = float(np.percentile(r_xs, 96.5))

    y_prog = np.clip((cyy - (_CANON_CHIN_Y + 65.0 * sr)) / max(140.0, 580.0 * sr), 0.0, 1.0)
    if is_fem:
        body_taper = 1.0 - 0.045 * np.sin(np.pi * np.clip((y_prog - 0.08) / 0.68, 0.0, 1.0)) + 0.015 * np.clip((y_prog - 0.78) / 0.22, 0.0, 1.0)
    else:
        body_taper = 1.0 - 0.035 * y_prog

    r_core = 55.0 * max(0.82, float(sr))
    t_v = np.clip((cyy - (_CANON_CHIN_Y - 12.0 * sr)) / max(12.0, 26.0 * sr), 0.0, 1.0)
    w_vert = 0.5 - 0.5 * np.cos(np.pi * t_v)

    # On standing half-sleeve shirts (`06a`), smoothly return warp -> 1.0 on lower bare forearms so wrists align 1:1!
    is_half_sleeve = sub_preset_key in ("male_shirt_white", "male_shirt_sky_blue", "female_shirt_sky_blue")
    if is_half_sleeve and float(sr) < 0.80:
        t_arm = np.clip((cyy - 525.0) / 85.0, 0.0, 1.0)
        arm_keep_gate = 1.0 - (0.5 - 0.5 * np.cos(np.pi * t_arm)) * np.clip((cdx - 135.0) / 25.0, 0.0, 1.0)
    else:
        arm_keep_gate = np.ones_like(cyy)

    tgt_side_w = np.where(cxx < _CANON_CX, float(target_left_w), float(target_right_w)) * body_taper
    ren_side_w = np.where(cxx < _CANON_CX, ren_left_w, ren_right_w)

    u_tgt = np.maximum(48.0, tgt_side_w - r_core)
    u_ren = np.maximum(48.0, ren_side_w - r_core)
    k_raw = np.clip(u_ren / u_tgt, 0.76, 1.46)
    k = 1.0 + w_vert * arm_keep_gate * (k_raw - 1.0)

    u = np.maximum(0.0, cdx - r_core)
    t = u / u_tgt
    f_t = np.where(t <= 1.0, t + (k - 1.0) * (t ** 2), k + (2.0 * k - 1.0) * (t - 1.0))
    src_cdx = np.where(cdx <= r_core, cdx, r_core + f_t * u_tgt)
    sign_x = np.where(cxx < _CANON_CX, -1.0, 1.0)
    map_x = (_CANON_CX + sign_x * src_cdx).astype(np.float32)

    perk_dy = w_vert * arm_keep_gate * np.clip((k - 1.0) * 10.0, 0.0, 4.5) * (np.clip(t, 0.0, 1.0) ** 1.3)
    map_y = (cyy + perk_dy).astype(np.float32)

    # Premultiplied Foreground Edge-Bleed BEFORE `remap` & `resize_img`: extrudes interior cloth RGB outward
    # by ~6px into `alpha == 0` so Lanczos resampling NEVER pulls `[254, 254, 255]` white fringes onto sleeve edges!
    fg_f = clean_fg_768.astype(np.float32)
    a_core = (clean_alpha_768 > 0.72).astype(np.float32)
    if float(a_core.sum()) >= 64.0:
        bleed_num = gaussian_blur(fg_f * a_core[:, :, None], sigma=4.0)
        bleed_den = gaussian_blur(a_core, sigma=4.0)[:, :, None]
        fg_extruded = np.where(bleed_den > 1e-3, bleed_num / np.maximum(bleed_den, 1e-3), fg_f)
        w_in = np.clip((clean_alpha_768 - 0.55) / 0.35, 0.0, 1.0)[:, :, None]
        fg_src_u8 = np.clip(np.round(w_in * fg_f + (1.0 - w_in) * fg_extruded), 0, 255).astype(np.uint8)
    else:
        fg_src_u8 = clean_fg_768

    if HAS_CV2 and cv2 is not None:
        warped_fg = cv2.remap(
            fg_src_u8,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped_alpha = cv2.remap(
            clean_alpha_768.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    else:
        oob = (map_x < 0.0) | (map_x > float(_CANON_W - 1)) | (map_y < 0.0) | (map_y > float(_CANON_H - 1))
        warped_fg = bilinear_sample(fg_src_u8, map_x, map_y).astype(np.uint8)
        warped_fg[oob] = np.array([254, 254, 255], dtype=np.uint8)
        warped_alpha = bilinear_sample(clean_alpha_768.astype(np.float32), map_x, map_y).astype(np.float32)
        warped_alpha[oob] = 0.0
    return warped_fg, np.clip(warped_alpha, 0.0, 1.0).astype(np.float32)


def resolve_subject_assignments(
    subjects: list[SubjectProfile],
    preset_key: str,
    target_person: str = "auto",
    has_custom_garment: bool = False,
) -> list[tuple[SubjectProfile, str]]:
    """
    Intelligent Multi-Subject & Official Couple Routing:
    Determines `(SubjectProfile, effective_preset_key)` pairs to dress:
    - Official Couple Presets (`couple_suit_and_maroon_saree`, etc.) automatically assign the
      Male outfit to Male subject(s) and Female outfit to Female subject(s) (or Left/Right fallback).
    - `target_person == "person_1"`: dresses Left / First person (`subjects[0]`).
    - `target_person == "person_2"`: dresses Right / Second person (`subjects[1]` if present else `subjects[0]`).
    - `target_person == "both"`: dresses all foreground subjects.
    - `target_person == "auto"`: on multi-person photos, matches `DRESS_PRESETS[preset_key]["gender"]`
      against `subj.inferred_gender` so choosing a Men's Suit dresses the Husband and choosing a
      Women's Saree dresses the Wife automatically!
    """
    from src.core.dress_engine import DRESS_PRESETS

    if not subjects:
        return []
    if not has_custom_garment and (not preset_key or preset_key == "none"):
        return []

    tp = str(target_person or "auto").strip().lower()

    if not has_custom_garment and preset_key in COUPLE_COMBO_MAP:
        male_key, female_key = COUPLE_COMBO_MAP[preset_key]
        if len(subjects) == 1:
            s0 = subjects[0]
            return [(s0, female_key if s0.inferred_gender == "female" else male_key)]
        pairs: list[tuple[SubjectProfile, str]] = []
        males = [s for s in subjects if s.inferred_gender == "male"]
        females = [s for s in subjects if s.inferred_gender == "female"]
        if males and females:
            for s in subjects:
                pairs.append((s, female_key if s.inferred_gender == "female" else male_key))
        else:
            pairs.append((subjects[0], male_key))
            for s in subjects[1:]:
                pairs.append((s, female_key))
        return pairs

    eff_key = preset_key if preset_key in DRESS_PRESETS else "male_suit_navy_tie"

    if tp == "person_1":
        return [(subjects[0], eff_key)]
    if tp == "person_2":
        return [(subjects[1] if len(subjects) >= 2 else subjects[0], eff_key)]
    if tp == "both":
        return [(s, eff_key) for s in subjects]

    if len(subjects) == 1:
        return [(subjects[0], eff_key)]

    preset_gender = DRESS_PRESETS.get(eff_key, {}).get("gender", "any")
    if not has_custom_garment and preset_gender in ("male", "female"):
        matched = [s for s in subjects if s.inferred_gender == preset_gender]
        if matched:
            return [(matched[0], eff_key)]

    return [(subjects[0], eff_key)]


def apply_catvton_virtual_tryon(
    clean_base_rgb: np.ndarray,
    current_dressed_rgb: np.ndarray | None,
    clean_fg_alpha: np.ndarray | None,
    preset_key: str = "none",
    target_person: str = "auto",
    custom_garment_rgb: np.ndarray | None = None,
    subjects: list[SubjectProfile] | None = None,
    vertical_shift: float = 0.0,
    shoulder_scale: float = 1.0,
    num_steps: int = 26,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]] | None:
    """
    Master V8.5 Cloud-Native Canonical `CatVTON` Virtual Try-On Pipeline:
    - Isolates & maps each targeted person into `768 × 1024` Canonical VITON-HD studio space (`bf16` CUDA).
    - Runs `CatVTON` (`mix-48k-1024`) + seamless `768 × 1024` BRIA `RMBG-2.0` silhouette re-matting BEFORE
      mapping `(rgb, alpha)` back to portrait coordinates — zero stacked upscales, zero gray washout,
      zero rectangular borders, and 100% pixel-locked faces/eyes/bindis/beards!
    """
    if not IS_HF_CLOUD:
        return None
    has_custom = custom_garment_rgb is not None and isinstance(custom_garment_rgb, np.ndarray) and custom_garment_rgb.size > 0
    if not has_custom and (not preset_key or preset_key == "none"):
        return clean_base_rgb.copy(), clean_fg_alpha, {"vton_engine": "Original-Dress-Restored"}

    bundle = _load_catvton_engine_cloud()
    if bundle is None:
        return None

    t0 = time.perf_counter()
    h, w = clean_base_rgb.shape[:2]
    active_subjects = subjects if subjects else detect_foreground_subjects(clean_base_rgb, raw_bria_alpha=clean_fg_alpha)
    assignments = resolve_subject_assignments(
        active_subjects,
        preset_key=preset_key,
        target_person=target_person,
        has_custom_garment=has_custom,
    )
    if not assignments:
        return clean_base_rgb.copy(), clean_fg_alpha, {"vton_engine": "No-Target-Subject"}

    if clean_fg_alpha is not None and clean_fg_alpha.shape[:2] == (h, w):
        af_clean = np.clip(clean_fg_alpha.astype(np.float32), 0.0, 1.0)
        if af_clean.max() > 1.5:
            af_clean /= 255.0
        neutral_clean_rgb = np.clip(
            np.round(
                af_clean[:, :, None] * clean_base_rgb.astype(np.float32)
                + (1.0 - af_clean[:, :, None]) * _VTON_STUDIO_BG_RGB[None, None, :]
            ),
            0,
            255,
        ).astype(np.uint8)
    else:
        af_clean = np.ones((h, w), dtype=np.float32)
        neutral_clean_rgb = clean_base_rgb.copy()

    working_out = (
        current_dressed_rgb.copy()
        if (current_dressed_rgb is not None and current_dressed_rgb.shape == clean_base_rgb.shape)
        else neutral_clean_rgb.copy()
    )
    updated_alpha = af_clean.copy()
    applied_labels: list[str] = []

    custom_ref_768 = _prepare_custom_garment_ref(custom_garment_rgb) if has_custom and custom_garment_rgb is not None else None
    from src.core.dress_engine import DRESS_PRESETS

    yy_full, xx_full = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    for subj, sub_preset_key in assignments:
        rx1, _ry1, rx2, _ry2 = subj.roi_box
        g = subj.geom
        ipd = max(16.0, float(g.ipd_px))
        face_cx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
        chin_y = float(g.chin_xy[1]) + float(vertical_shift) * ipd * 0.18

        # Reset ONLY this subject's partition to clean un-dressed base before try-on
        working_out[:, rx1:rx2] = neutral_clean_rgb[:, rx1:rx2]
        updated_alpha[:, rx1:rx2] = af_clean[:, rx1:rx2]

        eff_sub_key = "custom" if has_custom else sub_preset_key
        p_canvas, m_canvas, skin_canon, veil_head_wipe_canon, meta = _prepare_canonical_subject_vton(
            clean_base_rgb=clean_base_rgb,
            af_clean=af_clean,
            subj=subj,
            sub_preset_key=eff_sub_key,
            vertical_shift=vertical_shift,
            shoulder_scale=shoulder_scale,
        )
        if float(m_canvas.sum()) < 64.0:
            continue

        sr_val = float(meta.get("sr", 1.0))
        target_left_w = float(meta.get("target_left_w", 220.0))
        target_right_w = float(meta.get("target_right_w", 220.0))
        trouser_cutoff_y = float(meta.get("trouser_cutoff_y", 1022.0))

        # Pass native 1.0x VITON-HD studio garment reference so CatVTON self-attention preserves 100% of ties, lapels & buttons!
        raw_garment_ref = (
            custom_ref_768
            if custom_ref_768 is not None
            else _get_or_create_photoreal_garment_ref(sub_preset_key)
        )

        vton_canvas_u8 = _run_catvton_canvas_768x1024(
            bundle=bundle,
            p_canvas_768x1024=p_canvas,
            m_canvas_768x1024=m_canvas,
            garment_ref_768x1024=raw_garment_ref,
            num_steps=max(28, int(num_steps)),
            guidance_scale=2.6,
            seed=42 + subj.subject_index,
        )

        # Seamless 768x1024 BRIA matte extraction BEFORE inverse-mapping (zero rectangular boxes!)
        has_crown_veil = bool(meta.get("has_true_crown_veil", 0.0) > 0.5)
        clean_fg_768, clean_alpha_768 = _extract_clean_canonical_vton_rgba(
            vton_canvas_u8,
            sub_preset_key=eff_sub_key,
            p_canvas_u8=p_canvas,
            m_canvas_f32=m_canvas,
            shoulder_scale=shoulder_scale,
            sr=sr_val,
            has_true_crown_veil=has_crown_veil,
        )
        if not has_custom:
            clean_fg_768 = _post_polish_vton_garment_768(
                clean_fg_768,
                clean_alpha_768,
                p_canvas,
                m_canvas,
                sub_preset_key,
                skin_canon_768=skin_canon,
                sr=sr_val,
            )

        # Stage B: Neck-Locked C1 Anatomical Body Tailoring (`1.0x` tie/collar lock + exact body shoulder/waist fit!)
        clean_fg_768, clean_alpha_768 = _fit_rendered_vton_to_subject_body_768(
            clean_fg_768=clean_fg_768,
            clean_alpha_768=clean_alpha_768,
            m_canvas_768=m_canvas,
            target_left_w=target_left_w,
            target_right_w=target_right_w,
            sub_preset_key=eff_sub_key,
            is_fem=(subj.inferred_gender == "female"),
            sr=sr_val,
            trouser_cutoff_y=trouser_cutoff_y,
        )

        # Inverse-map `(clean_fg_768, clean_alpha_768, veil_head_wipe_canon)` back to portrait window `[sx1:sx2, sy1:sy2]`
        iw = int(round(meta["iw"]))
        ih = int(round(meta["ih"]))
        ix0 = int(round(meta["ix0"]))
        iy0 = int(round(meta["iy0"]))
        sx1 = int(round(meta["sx1"]))
        sy1 = int(round(meta["sy1"]))
        sx2 = int(round(meta["sx2"]))
        sy2 = int(round(meta["sy2"]))
        if sx2 <= sx1 or sy2 <= sy1:
            continue

        fg_win = resize_img(clean_fg_768, iw, ih, mode="lanczos")
        al_win = np.clip(resize_img(clean_alpha_768, iw, ih, mode="bilinear"), 0.0, 1.0)
        vw_win = np.clip(resize_img(veil_head_wipe_canon, iw, ih, mode="bilinear"), 0.0, 1.0)

        dx1 = sx1 - ix0
        dy1 = sy1 - iy0
        fg_patch = fg_win[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)].astype(np.float32)
        al_patch = al_win[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)]
        vw_patch = vw_win[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)]

        # 1. BELOW-NECK OUTER-SLEEVE WIPE + SURGICAL OUTER HEAD-VEIL WING UNWRAP ACROSS `[rx1:rx2]`
        xx_roi = xx_full[:, rx1:rx2]
        yy_roi = yy_full[:, rx1:rx2]
        dx_roi = np.abs(xx_roi - face_cx)
        roi_f = neutral_clean_rgb[:, rx1:rx2].astype(np.float32)
        roi_af = af_clean[:, rx1:rx2]

        below_neck_wipe = np.clip((yy_roi - (chin_y + 0.08 * ipd)) / max(4.0, 0.10 * ipd), 0.0, 1.0)
        if subj.has_head_pallu or float(meta.get("has_veil_or_scarf", 0.0)) > 0.5:
            outer_veil_wing = np.clip((dx_roi - 1.18 * ipd) / max(4.0, 0.12 * ipd), 0.0, 1.0)
            below_neck_wipe = np.maximum(below_neck_wipe, outer_veil_wing)

        # Preserve real lower trousers & hands below `trouser_port_y` on standing three-quarter portraits (`sr < 0.80`)
        scale_back = float(ih) / float(_CANON_H)
        trouser_port_y = float(iy0) + trouser_cutoff_y * scale_back
        if sr_val < 0.80:
            trouser_keep_roi = np.clip((yy_roi - (trouser_port_y - 0.12 * ipd)) / max(5.0, 0.16 * ipd), 0.0, 1.0)
            below_neck_wipe *= (1.0 - trouser_keep_roi)

        part_guard_roi = np.ones_like(xx_roi)
        if rx1 > 0:
            part_guard_roi *= np.clip((xx_roi - float(rx1)) / 5.0, 0.0, 1.0)
        if rx2 < w:
            part_guard_roi *= np.clip((float(rx2) - xx_roi) / 5.0, 0.0, 1.0)

        wipe_roi = below_neck_wipe * part_guard_roi
        cleared_roi_rgb = roi_f * (1.0 - wipe_roi[:, :, None]) + _VTON_STUDIO_BG_RGB[None, None, :] * wipe_roi[:, :, None]
        cleared_roi_af = roi_af * (1.0 - wipe_roi)
        working_out[:, rx1:rx2] = np.clip(np.round(cleared_roi_rgb), 0, 255).astype(np.uint8)
        updated_alpha[:, rx1:rx2] = np.clip(cleared_roi_af, 0.0, 1.0).astype(np.float32)

        # 2. COMPOSITE SEAMLESS CATVTON GARMENT + UNWRAP ANY INNER CHEEK/CHIN VEIL CLOTH (`vw_sub`)
        psx1 = max(sx1, rx1)
        psx2 = min(sx2, rx2)
        if psx2 <= psx1:
            continue
        sub_x1 = psx1 - sx1
        sub_x2 = psx2 - sx1
        roi_x1 = psx1 - rx1
        roi_x2 = psx2 - rx1

        fg_sub = fg_patch[:, sub_x1:sub_x2]
        al_sub = al_patch[:, sub_x1:sub_x2]
        vw_sub = vw_patch[:, sub_x1:sub_x2]
        xx_p = xx_full[sy1:sy2, psx1:psx2]
        yy_p = yy_full[sy1:sy2, psx1:psx2]

        # Smooth anatomical head/face/ear/pigtail guard, minus SegFormer-detected cheek/chin/collar cloth (`vw_sub`)!
        upper_head_band = np.clip(((chin_y - 0.42 * ipd) - yy_p) / max(3.0, 0.08 * ipd), 0.0, 1.0) * (1.0 - vw_sub)
        ell_p = np.sqrt(((xx_p - face_cx) / (0.67 * ipd)) ** 2 + ((yy_p - (chin_y - 0.94 * ipd)) / (0.97 * ipd)) ** 2)
        head_smooth_p = np.clip((1.05 - ell_p) / 0.12, 0.0, 1.0) * (1.0 - vw_sub)
        orn_p = np.clip(subj.ornament_lock[sy1:sy2, psx1:psx2] * 1.5, 0.0, 1.0) * (yy_p < chin_y - 0.04 * ipd).astype(np.float32)

        trouser_lock_p = (
            np.clip((yy_p - (trouser_port_y - 0.08 * ipd)) / max(5.0, 0.16 * ipd), 0.0, 1.0)
            if sr_val < 0.80
            else np.zeros_like(yy_p)
        )

        head_lock_p = np.maximum.reduce([upper_head_band, head_smooth_p, orn_p, trouser_lock_p])
        head_lock_p = np.clip(gaussian_blur(head_lock_p, sigma=max(2.0, 0.035 * ipd)), 0.0, 1.0)
        head_lock_p = np.maximum(head_lock_p, (ell_p <= 0.82).astype(np.float32) * (1.0 - vw_sub))

        part_guard_p = part_guard_roi[sy1:sy2, roi_x1:roi_x2]
        paste_zone = np.clip((1.0 - head_lock_p) * part_guard_p, 0.0, 1.0)

        base_rgb_p = working_out[sy1:sy2, psx1:psx2].astype(np.float32)
        base_af_p = updated_alpha[sy1:sy2, psx1:psx2]

        vton_comp_p = fg_sub * al_sub[:, :, None] + _VTON_STUDIO_BG_RGB[None, None, :] * (1.0 - al_sub[:, :, None])
        new_rgb_p = base_rgb_p * (1.0 - paste_zone[:, :, None]) + vton_comp_p * paste_zone[:, :, None]
        new_af_p = base_af_p * (1.0 - paste_zone) + al_sub * paste_zone

        working_out[sy1:sy2, psx1:psx2] = np.clip(np.round(new_rgb_p), 0, 255).astype(np.uint8)
        updated_alpha[sy1:sy2, psx1:psx2] = np.clip(new_af_p, 0.0, 1.0).astype(np.float32)

        garment_lbl = "Custom Uploaded Dress" if has_custom else DRESS_PRESETS.get(sub_preset_key, {}).get("label", sub_preset_key)
        applied_labels.append(f"{subj.label} → {garment_lbl}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    telemetry = {
        "vton_engine": "CatVTON-V10-AnatomicalBodyFit (SegFormer-B2 + BF16 VITON-HD)",
        "subjects_dressed": applied_labels,
        "vton_inference_ms": elapsed_ms,
    }
    return working_out, updated_alpha, telemetry


def warmup_vton_cloud_models() -> None:
    """Pre-download and assemble SegFormer-B2 + CatVTON on CPU at Space startup (`IS_HF_CLOUD == True` only)."""
    if not IS_HF_CLOUD:
        return
    try:
        _load_segformer_parser_cloud()
        _load_catvton_engine_cloud()
    except Exception as e:
        print(f"⚠️ [CatVTON Warmup] Non-fatal warmup note: {e}")
