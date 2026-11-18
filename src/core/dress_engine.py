"""
SartorialFit™ — Anthropometric Female & Male Dress Change Engine.
Engineered with mathematical precision for Indian & International Passport Photos:
1. Cervical-Clavicular Skeletal Projection from Facial Anthropometrics (IPD, Chin, Roll Angle).
2. Biharmonic Thin-Plate Spline (TPS, Delta^2 f = 0, U(r) = r^2 ln r) 9-Anchor Garment Warping.
3. Cervical-Décolletage 3D Cylindrical Skin Chromophore Continuity (CIE L*a*b* matched to cheek/jaw).
4. Female Long-Hair & Jewelry Occlusion Preservation Matting (Guided-Filter strand alpha so female
   hair drapes naturally OVER the new saree/kurti/blazer).
5. Ambient-Occlusion Collar Shadowing & Multi-Scale Guided Boundary Fusion.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
from PIL import Image
from src.shared.config import BASE_DATA_DIR
from src.core.cv_ops import (
    gaussian_blur,
    guided_filter,
    rgb_to_lab,
    lab_to_rgb,
    warp_image_tps,
)
from src.core.facial_math_engine import FacialAnthropometrics, analyze_face_geometry

GARMENTS_DIR = os.path.join(BASE_DATA_DIR, "garments")


# ============================================================================
# 1. CATALOG OF FEMALE & MALE STUDIO GARMENT PRESETS
# ============================================================================

DRESS_PRESETS: dict[str, dict] = {
    "none": {
        "label": "🚫 Keep Original Dress",
        "gender": "any",
        "style": "none",
    },
    # --- FEMALE TRADITIONAL & FORMAL ATTIRE ---
    "female_saree_maroon_gold": {
        "label": "👩🏻 Women: Maroon Silk Saree (Gold Border)",
        "gender": "female",
        "style": "saree",
        "primary_rgb": (136, 22, 42),       # Rich Banarasi Maroon
        "secondary_rgb": (110, 16, 32),     # Tailored Maroon Blouse
        "border_rgb": (228, 184, 74),       # 24K Gold Zari Border
        "neckline": "scoop",
    },
    "female_saree_royal_blue": {
        "label": "👩🏻 Women: Royal Blue Silk Saree (Gold Border)",
        "gender": "female",
        "style": "saree",
        "primary_rgb": (24, 58, 138),       # Royal Kanjeevaram Blue
        "secondary_rgb": (18, 42, 108),     # Matching Silk Blouse
        "border_rgb": (232, 192, 88),       # Gold Zari
        "neckline": "scoop",
    },
    "female_salwar_emerald": {
        "label": "👩🏻 Women: Emerald Green Salwar Suit",
        "gender": "female",
        "style": "salwar",
        "primary_rgb": (18, 98, 72),        # Emerald Green Kameez
        "secondary_rgb": (152, 28, 56),     # Contrast Crimson/Gold Dupatta
        "border_rgb": (224, 186, 84),       # Zari trim
        "neckline": "v_modest",
    },
    "female_kurti_ivory_gold": {
        "label": "👩🏻 Women: Classic Ivory & Gold Kurti",
        "gender": "female",
        "style": "kurti",
        "primary_rgb": (242, 236, 222),     # Warm Ivory Silk/Cotton
        "secondary_rgb": (218, 182, 98),    # Gold Chikankari Embroidery Collar
        "border_rgb": (196, 156, 72),
        "neckline": "notch_collar",
    },
    "female_blazer_navy": {
        "label": "👩🏻‍💼 Women: Navy Blazer & White Blouse",
        "gender": "female",
        "style": "female_blazer",
        "primary_rgb": (22, 34, 62),        # Tailored Navy Blazer
        "secondary_rgb": (246, 248, 252),   # Crisp White Inner Silk Blouse
        "border_rgb": (38, 54, 92),         # Satin Lapel highlight
        "neckline": "blazer_blouse",
    },
    "female_blazer_charcoal": {
        "label": "👩🏻‍💼 Women: Charcoal Suit & Cream Blouse",
        "gender": "female",
        "style": "female_blazer",
        "primary_rgb": (40, 42, 48),        # Charcoal Suit Jacket
        "secondary_rgb": (248, 244, 236),   # Soft Cream Inner Blouse
        "border_rgb": (62, 65, 74),
        "neckline": "blazer_blouse",
    },
    "female_uniform_navy_white": {
        "label": "👩🏻 Girls / Women: White & Navy V-Yoke School/College Uniform",
        "gender": "female",
        "style": "female_shirt",
        "primary_rgb": (244, 246, 250),     # Crisp White Pin-Dot Cotton Kameez
        "secondary_rgb": (24, 42, 84),      # Symmetrical Navy Blue V-Yoke Overlay
        "border_rgb": (24, 42, 84),         # Navy Cuff Band
        "neckline": "collar_v",
    },
    "female_shirt_sky_blue": {
        "label": "👩🏻‍💼 Women: Sky-Blue Formal Shirt",
        "gender": "female",
        "style": "female_shirt",
        "primary_rgb": (176, 210, 238),     # Official Passport Sky Blue
        "secondary_rgb": (240, 246, 252),   # Crisp Inner Collar Trim
        "border_rgb": (145, 185, 220),
        "neckline": "collar_v",
    },
    # --- MALE EXECUTIVE & FORMAL ATTIRE ---
    "male_suit_navy_tie": {
        "label": "👨🏻‍💼 Men: Navy Blue Suit & Red Tie",
        "gender": "male",
        "style": "male_suit",
        "primary_rgb": (20, 32, 58),        # Navy Suit Jacket
        "secondary_rgb": (248, 250, 254),   # White Dress Shirt
        "border_rgb": (156, 24, 42),        # Burgundy Silk Tie
        "neckline": "suit_tie",
    },
    "male_suit_charcoal_tie": {
        "label": "👨🏻‍💼 Men: Black Suit & Blue Tie",
        "gender": "male",
        "style": "male_suit",
        "primary_rgb": (32, 34, 38),        # Charcoal Black Jacket
        "secondary_rgb": (248, 250, 254),   # White Dress Shirt
        "border_rgb": (28, 68, 148),        # Royal Blue Silk Tie
        "neckline": "suit_tie",
    },
    "male_blazer_royal": {
        "label": "👨🏻‍💼 Men: Navy Blazer & White Shirt (No Tie)",
        "gender": "male",
        "style": "male_blazer",
        "primary_rgb": (26, 42, 76),
        "secondary_rgb": (246, 248, 252),
        "border_rgb": (44, 64, 108),
        "neckline": "open_collar",
    },
    "male_shirt_white": {
        "label": "👨🏻 Men / Boys: Crisp White Formal Half-Sleeve Shirt",
        "gender": "male",
        "style": "male_shirt",
        "primary_rgb": (244, 246, 250),
        "secondary_rgb": (222, 228, 238),
        "border_rgb": (200, 208, 220),
        "neckline": "shirt_collar",
    },
    "male_shirt_sky_blue": {
        "label": "👨🏻 Men: Sky-Blue Formal Shirt",
        "gender": "male",
        "style": "male_shirt",
        "primary_rgb": (168, 206, 238),
        "secondary_rgb": (235, 244, 252),
        "border_rgb": (138, 180, 218),
        "neckline": "shirt_collar",
    },
    # --- OFFICIAL DUAL / COUPLE JOINT PHOTO ATTIRE (AUTO HUSBAND + WIFE) ---
    "couple_suit_and_maroon_saree": {
        "label": "👫 Official Couple: Men's Navy Suit + Women's Maroon Silk Saree",
        "gender": "couple",
        "style": "couple_combo",
        "primary_rgb": (20, 32, 58),
        "secondary_rgb": (136, 22, 42),
        "border_rgb": (228, 184, 74),
        "neckline": "couple",
    },
    "couple_blazer_and_royal_saree": {
        "label": "👫 Official Couple: Men's Navy Blazer + Women's Royal Blue Saree",
        "gender": "couple",
        "style": "couple_combo",
        "primary_rgb": (26, 42, 76),
        "secondary_rgb": (24, 58, 138),
        "border_rgb": (232, 192, 88),
        "neckline": "couple",
    },
    "couple_formal_shirts": {
        "label": "👫 Official Couple: Formal White & Sky-Blue Office Attire",
        "gender": "couple",
        "style": "couple_combo",
        "primary_rgb": (244, 246, 250),
        "secondary_rgb": (176, 210, 238),
        "border_rgb": (200, 208, 220),
        "neckline": "couple",
    },
    "couple_school_uniforms": {
        "label": "👫 Uniform Duo: Crisp White Shirt + Girls' Navy-V Uniform",
        "gender": "couple",
        "style": "couple_combo",
        "primary_rgb": (244, 246, 250),
        "secondary_rgb": (24, 42, 84),
        "border_rgb": (24, 42, 84),
        "neckline": "couple",
    },
}


# ============================================================================
# 2. PROCEDURAL 4K STUDIO GARMENT SYNTHESIS WITH FABRIC MICRO-WEAVE & SHADING
# ============================================================================

_TEMPLATE_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_HAIR_MATTE_CACHE: dict[str, object] = {"key": None, "matte": None}


def build_canonical_garment_template(
    preset_key: str,
    canvas_w: int = 900,
    canvas_h: int = 700,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a canonical high-resolution RGBA garment template + 9 anthropometric control points.
    Canonical coordinate system (canvas_w=900, canvas_h=700):
    - Neck Top Left / Right:   (350, 70),  (550, 70)
    - Neck Base Left / Right:  (320, 175), (580, 175)
    - Jugular Notch J:         (450, 235)
    - Shoulder Left / Right:   (105, 255), (795, 255)
    - Bottom Torso Left/Right: (65, 690),  (835, 690)
    Returns (garment_rgb, garment_alpha, src_landmarks_9x2).
    """
    cache_key = (str(preset_key), int(canvas_w), int(canvas_h))
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], cached[2]

    preset = DRESS_PRESETS.get(preset_key, DRESS_PRESETS["female_saree_maroon_gold"])
    style = preset["style"]
    c_prim = np.array(preset.get("primary_rgb", (136, 22, 42)), dtype=np.float32)
    c_sec = np.array(preset.get("secondary_rgb", (110, 16, 32)), dtype=np.float32)
    c_bord = np.array(preset.get("border_rgb", (228, 184, 74)), dtype=np.float32)

    yy, xx = np.meshgrid(
        np.arange(canvas_h, dtype=np.float32),
        np.arange(canvas_w, dtype=np.float32),
        indexing="ij",
    )
    cx = canvas_w * 0.5

    # Canonical 9 Anthropometric Landmarks
    bot_hem_y = float(canvas_h - 10.0)
    src_pts = np.array([
        [350.0,  70.0],     # 0: Neck Top Left
        [550.0,  70.0],     # 1: Neck Top Right
        [320.0, 175.0],     # 2: Neck Base Left
        [580.0, 175.0],     # 3: Neck Base Right
        [450.0, 235.0],     # 4: Jugular Notch J
        [105.0, 255.0],     # 5: Shoulder Left A_L
        [795.0, 255.0],     # 6: Shoulder Right A_R
        [ 65.0, bot_hem_y], # 7: Bottom Left Hem
        [835.0, bot_hem_y], # 8: Bottom Right Hem
    ], dtype=np.float32)

    # 1. Unified C2-Smooth Collar-to-Shoulder Top Contour & Wide Deltoid Torso Silhouette
    dx_abs = np.abs(xx - cx)
    max_half_w = 436.0 + 0.04 * np.maximum(0.0, yy - 250.0)
    outer_side_alpha = np.clip((max_half_w - dx_abs) / 6.0, 0.0, 1.0)

    # 2. 3D Cylindrical & Clavicular Normal-Mapped Studio Shading + Anisotropic Fabric Micro-Textures
    t_sh_base = np.clip((dx_abs - 104.0) / 241.0, 0.0, 1.0)
    base_shoulder_y = 162.0 + 93.0 * (t_sh_base ** 1.16) + 54.0 * np.maximum(0.0, (dx_abs - 345.0) / 95.0) ** 1.65
    nx_torso = np.clip((xx - cx) / 395.0, -1.0, 1.0)
    nz_torso = np.sqrt(np.maximum(0.12, 1.0 - 0.72 * nx_torso * nx_torso))
    clavicle_spec = 0.06 * np.exp(-0.5 * ((yy - (base_shoulder_y + 26.0)) / 24.0) ** 2) * (dx_abs > 125.0)
    cyl_shade = 0.78 + 0.26 * nz_torso + clavicle_spec
    vert_shade = 1.04 - 0.14 * np.clip((yy - 165.0) / 520.0, 0.0, 1.0)

    # Anisotropic textile micro-weaves (worsted wool twill, crisp cotton poplin, silk sheen)
    wool_twill = 1.0 + 0.018 * np.sin((xx + yy) * 1.65) + 0.008 * np.cos((xx - yy) * 2.35)
    cotton_poplin = 1.0 + 0.012 * (np.sin(xx * 2.15) * np.cos(yy * 2.15))
    silk_sheen = 1.0 + 0.022 * np.sin(xx * 1.45 + yy * 0.55) * np.cos(yy * 1.35 - xx * 0.35)
    weave = (
        wool_twill
        if style in ("male_suit", "male_blazer", "female_blazer")
        else (cotton_poplin if style in ("male_shirt", "female_shirt") else silk_sheen)
    )
    shade_3d = (cyl_shade * vert_shade * weave)[:, :, None]

    rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    if style == "saree":
        # Female Traditional Silk Saree + Fitted Blouse + Diagonal Pleated Pallu over Left Shoulder (viewer's right)
        peak_dx = 122.0
        peak_y = 166.0
        inner_neck_y = 246.0 - (246.0 - peak_y) * np.clip((dx_abs / peak_dx) ** 1.65, 0.0, 1.0)
        t_sh = np.clip((dx_abs - peak_dx) / (345.0 - peak_dx), 0.0, 1.0)
        outer_sh_y = peak_y + 89.0 * (t_sh ** 1.18) + 54.0 * np.maximum(0.0, (dx_abs - 345.0) / 95.0) ** 1.65
        top_contour_y = np.where(dx_abs <= peak_dx, inner_neck_y, outer_sh_y) + 4.0 * np.exp(
            -0.5 * ((dx_abs - peak_dx) / 10.0) ** 2
        )
        alpha = np.clip((yy - top_contour_y) / 4.5, 0.0, 1.0) * outer_side_alpha

        # Base blouse color
        rgb[:, :, :] = c_sec[None, None, :] * shade_3d

        # Gold pipings along blouse scoop neckline
        neck_dist = np.abs(yy - inner_neck_y)
        blouse_trim = np.exp(-0.5 * (neck_dist / 5.0) ** 2) * (dx_abs < peak_dx + 8.0)
        rgb = rgb * (1.0 - blouse_trim[:, :, None]) + (c_bord * 1.08)[None, None, :] * blouse_trim[:, :, None]

        # Diagonal Saree Pallu sweeping from lower-left up to right shoulder
        pallu_edge_x = 532.0 - 0.72 * (yy - 175.0)
        pallu_mask = np.clip((xx - pallu_edge_x) / 8.0, 0.0, 1.0)

        # Diagonal 3D silk pleats along the pallu
        pleat_coord = (xx - pallu_edge_x) * 0.80 + (yy - 175.0) * 0.35
        pleat_wave = 0.88 + 0.16 * np.sin(pleat_coord * 0.14) + 0.05 * np.cos(pleat_coord * 0.28)
        saree_body = c_prim[None, None, :] * shade_3d * pleat_wave[:, :, None]

        # Rich Gold/Silver Zari Border along the upper diagonal edge of the pallu
        zari_pattern = 0.86 + 0.22 * (np.sin(pleat_coord * 0.65) * np.cos((yy - xx) * 0.45))
        zari_rgb = c_bord[None, None, :] * zari_pattern[:, :, None] * cyl_shade[:, :, None]

        border_w = np.exp(-0.5 * (((xx - pallu_edge_x) - 24.0) / 16.5) ** 2)
        saree_with_border = saree_body * (1.0 - 0.88 * border_w[:, :, None]) + zari_rgb * (0.88 * border_w[:, :, None])

        # Drop shadow cast by saree pallu onto blouse underneath
        pallu_shadow = np.exp(-0.5 * ((xx - pallu_edge_x + 7.0) / 11.0) ** 2) * (xx < pallu_edge_x)
        rgb *= (1.0 - 0.28 * pallu_shadow[:, :, None])
        rgb = rgb * (1.0 - pallu_mask[:, :, None]) + saree_with_border * pallu_mask[:, :, None]

    elif style in ("salwar", "kurti"):
        # Female Formal Salwar Kameez / Kurti with Graceful V-Scoop Embroidered Neckline & Optional Dupatta
        peak_dx = 116.0
        peak_y = 164.0
        inner_neck_y = 246.0 - (246.0 - peak_y) * np.clip((dx_abs / peak_dx) ** 1.45, 0.0, 1.0)
        t_sh = np.clip((dx_abs - peak_dx) / (345.0 - peak_dx), 0.0, 1.0)
        outer_sh_y = peak_y + 91.0 * (t_sh ** 1.18) + 54.0 * np.maximum(0.0, (dx_abs - 345.0) / 95.0) ** 1.65
        top_contour_y = np.where(dx_abs <= peak_dx, inner_neck_y, outer_sh_y) + 4.0 * np.exp(
            -0.5 * ((dx_abs - peak_dx) / 10.0) ** 2
        )
        alpha = np.clip((yy - top_contour_y) / 4.5, 0.0, 1.0) * outer_side_alpha

        rgb[:, :, :] = c_prim[None, None, :] * shade_3d

        # Embroidered Placket & Neckline Border
        neck_dist = np.abs(yy - inner_neck_y)
        emb_band = (np.exp(-0.5 * (neck_dist / 12.0) ** 2) * (dx_abs < peak_dx + 12.0)) + (
            np.exp(-0.5 * (dx_abs / 18.0) ** 2) * ((yy >= inner_neck_y) & (yy < 490.0))
        )
        emb_band = np.clip(emb_band, 0.0, 1.0)
        emb_tex = 0.86 + 0.22 * np.sin(xx * 0.55) * np.cos(yy * 0.55)
        rgb = rgb * (1.0 - 0.80 * emb_band[:, :, None]) + (
            c_bord[None, None, :] * emb_tex[:, :, None] * (0.80 * emb_band[:, :, None])
        )

        if style == "salwar":
            dup_w = np.clip((dx_abs - (155.0 - 0.14 * (yy - 180.0))) / 12.0, 0.0, 1.0) * np.clip(
                (318.0 - dx_abs) / 12.0, 0.0, 1.0
            )
            dup_pleats = 0.89 + 0.14 * np.sin(dx_abs * 0.22)
            dup_rgb = c_sec[None, None, :] * shade_3d * dup_pleats[:, :, None]
            inner_edge = np.exp(-0.5 * ((dx_abs - (165.0 - 0.14 * (yy - 180.0))) / 9.0) ** 2)
            dup_rgb = dup_rgb * (1.0 - 0.82 * inner_edge[:, :, None]) + c_bord[None, None, :] * (
                0.82 * inner_edge[:, :, None]
            )
            rgb = rgb * (1.0 - dup_w[:, :, None]) + dup_rgb * dup_w[:, :, None]

    elif style == "female_blazer":
        # Women's Corporate Tailored Blazer + Inner Silk V-Blouse strictly contained inside Blazer Lapels
        peak_dx = 126.0
        peak_y = 142.0
        inner_neck_y = 244.0 - (244.0 - peak_y) * np.clip((dx_abs / peak_dx) ** 1.35, 0.0, 1.0)
        t_sh = np.clip((dx_abs - peak_dx) / (345.0 - peak_dx), 0.0, 1.0)
        outer_sh_y = peak_y + 113.0 * (t_sh ** 1.18) + 54.0 * np.maximum(0.0, (dx_abs - 345.0) / 95.0) ** 1.65
        top_contour_y = np.where(dx_abs <= peak_dx, inner_neck_y, outer_sh_y) + 5.0 * np.exp(
            -0.5 * ((dx_abs - peak_dx) / 11.0) ** 2
        )
        alpha = np.clip((yy - top_contour_y) / 4.5, 0.0, 1.0) * outer_side_alpha

        # Inner blouse V-zone is strictly inside peak_dx (108 < 126) so blouse never sticks out past jacket
        v_half_width = np.maximum(0.0, 108.0 - 0.25 * np.maximum(0.0, yy - peak_y))
        blouse_w = np.clip((v_half_width - dx_abs) / 4.0, 0.0, 1.0)

        jacket_rgb = c_prim[None, None, :] * shade_3d
        blouse_rgb = c_sec[None, None, :] * (cyl_shade * vert_shade * silk_sheen)[:, :, None] * 0.98

        # Tailored Satin Lapels with 3D bevel highlight, pick-stitching, and cast shadow onto blouse
        lapel_dist = dx_abs - v_half_width
        lapel_mask = np.exp(-0.5 * ((lapel_dist - 22.0) / 16.0) ** 2) * ((lapel_dist >= 0.0) & (yy < canvas_h - 40.0))
        lapel_seam = np.exp(-0.5 * ((lapel_dist - 46.0) / 5.5) ** 2) * (yy < canvas_h - 40.0)
        pick_stitch = (
            np.exp(-0.5 * ((lapel_dist - 41.0) / 1.5) ** 2)
            * (np.sin(yy * 0.9) > 0.2)
            * (yy < canvas_h - 50.0)
        )
        inner_cast = np.exp(-0.5 * ((v_half_width - dx_abs) / 9.0) ** 2) * (dx_abs <= v_half_width)
        blouse_rgb *= (1.0 - 0.18 * inner_cast[:, :, None])
        jacket_rgb = jacket_rgb * (
            1.0 + 0.15 * lapel_mask[:, :, None] - 0.15 * lapel_seam[:, :, None] - 0.05 * pick_stitch[:, :, None]
        )

        rgb = jacket_rgb * (1.0 - blouse_w[:, :, None]) + blouse_rgb * blouse_w[:, :, None]

    elif style in ("female_shirt", "male_shirt"):
        # Crisp Official Collared Shirt with Natural Collar Stand, Pick-Stitching & Folded Spread Points
        peak_dx = 122.0
        peak_y = 144.0
        v_depth = 224.0 if style == "female_shirt" else 208.0
        inner_neck_y = v_depth - (v_depth - peak_y) * np.clip((dx_abs / peak_dx) ** 1.32, 0.0, 1.0)
        t_sh = np.clip((dx_abs - peak_dx) / (345.0 - peak_dx), 0.0, 1.0)
        outer_sh_y = peak_y + 111.0 * (t_sh ** 1.18) + 54.0 * np.maximum(0.0, (dx_abs - 345.0) / 95.0) ** 1.65
        top_contour_y = np.where(dx_abs <= peak_dx, inner_neck_y, outer_sh_y) + 4.5 * np.exp(
            -0.5 * ((dx_abs - peak_dx) / 11.0) ** 2
        )
        alpha = np.clip((yy - top_contour_y) / 4.0, 0.0, 1.0) * outer_side_alpha

        rgb[:, :, :] = c_prim[None, None, :] * shade_3d

        # 3D Folded Pointed Shirt Collar Wings wrapping from collar stand down to collar points
        collar_outer_w = 160.0 - 0.12 * np.maximum(0.0, yy - peak_y)
        collar_inner_w = np.maximum(0.0, peak_dx * (1.0 - (yy - peak_y) / max(10.0, v_depth - peak_y)))
        collar_bot_y = v_depth + 22.0 + 0.22 * dx_abs
        in_collar = (yy >= peak_y) & (yy <= collar_bot_y) & (dx_abs >= collar_inner_w) & (dx_abs <= collar_outer_w)
        collar_edge = (
            np.exp(-0.5 * ((dx_abs - collar_outer_w) / 4.0) ** 2) * (yy <= collar_bot_y)
            + np.exp(-0.5 * ((yy - collar_bot_y) / 4.0) ** 2) * (dx_abs <= collar_outer_w)
        )
        collar_roll = np.exp(-0.5 * ((dx_abs - (collar_inner_w + 18.0)) / 14.0) ** 2) * in_collar
        rgb = np.where(in_collar[:, :, None], rgb * (1.05 + 0.05 * collar_roll[:, :, None]), rgb)
        rgb *= (1.0 - 0.18 * np.clip(collar_edge, 0.0, 1.0)[:, :, None])

        # Center button placket below V-point
        placket = (dx_abs <= 16.0) & (yy > v_depth)
        placket_edge = np.exp(-0.5 * ((dx_abs - 16.0) / 3.0) ** 2) * (yy > v_depth)
        rgb = np.where(placket[:, :, None], rgb * 1.03, rgb)
        rgb *= (1.0 - 0.10 * placket_edge[:, :, None])
        for btn_y in (320.0, 445.0, 570.0):
            r_btn = np.hypot(xx - cx, yy - btn_y)
            btn = r_btn <= 7.5
            btn_rim = (r_btn >= 5.5) & (r_btn <= 7.5)
            rgb = np.where(btn[:, :, None], np.array([242.0, 244.0, 248.0])[None, None, :], rgb)
            rgb = np.where(btn_rim[:, :, None], np.array([195.0, 202.0, 212.0])[None, None, :], rgb)

        if preset_key == "female_uniform_navy_white":
            # Symmetrical Navy Blue V-Yoke School/College Uniform Overlay + Micro Pin-Dot Kameez
            pindot = (((np.round(xx) % 12) == 0) & ((np.round(yy) % 12) == 0)).astype(np.float32) * (1.0 - in_collar.astype(np.float32))
            rgb = rgb * (1.0 - 0.28 * pindot[:, :, None]) + c_sec[None, None, :] * (0.28 * pindot[:, :, None])
            yoke_inner = np.maximum(0.0, 128.0 - 0.34 * np.maximum(0.0, yy - 165.0))
            yoke_outer = 268.0 - 0.24 * np.maximum(0.0, yy - 180.0)
            yoke_mask = (
                np.clip((dx_abs - yoke_inner) / 3.5, 0.0, 1.0)
                * np.clip((yoke_outer - dx_abs) / 3.5, 0.0, 1.0)
                * np.clip((525.0 - yy) / 4.0, 0.0, 1.0)
                * (1.0 - in_collar.astype(np.float32))
            )
            rgb = rgb * (1.0 - yoke_mask[:, :, None]) + (c_sec[None, None, :] * shade_3d) * yoke_mask[:, :, None]

    else:
        # Male Executive Suit / Blazer + Tailored Spread Shirt Collar + 3D Windsor Silk Tie + Notched Lapels
        peak_dx = 132.0
        peak_y = 144.0
        center_collar_y = 180.0 if style == "male_suit" else 200.0
        inner_neck_y = center_collar_y - (center_collar_y - peak_y) * np.clip((dx_abs / peak_dx) ** 1.42, 0.0, 1.0)
        t_sh = np.clip((dx_abs - peak_dx) / (345.0 - peak_dx), 0.0, 1.0)
        outer_sh_y = peak_y + 111.0 * (t_sh ** 1.18) + 54.0 * np.maximum(0.0, (dx_abs - 345.0) / 95.0) ** 1.65
        top_contour_y = np.where(dx_abs <= peak_dx, inner_neck_y, outer_sh_y) + 3.5 * np.exp(
            -0.5 * ((dx_abs - peak_dx) / 10.5) ** 2
        )
        alpha = np.clip((yy - top_contour_y) / 4.0, 0.0, 1.0) * outer_side_alpha

        # White dress shirt V-zone wraps around the natural neck width (114.0) while staying strictly inside
        # the suit jacket lapels (114.0 < peak_dx=132.0), so the navy suit jacket collar wraps smoothly outside!
        v_half_width = np.maximum(0.0, 114.0 - 0.275 * np.maximum(0.0, yy - peak_y))
        shirt_w = np.clip((v_half_width - dx_abs) / 3.5, 0.0, 1.0)

        suit_rgb = c_prim[None, None, :] * shade_3d
        # Crisp studio-white dress shirt with cotton poplin micro-weave & 3D cylindrical studio lighting
        shirt_shade = (0.90 + 0.07 * nz_torso) * (1.00 - 0.08 * np.clip((yy - 165.0) / 520.0, 0.0, 1.0)) * cotton_poplin
        shirt_rgb = np.clip(c_sec[None, None, :] * shirt_shade[:, :, None], 0.0, 255.0)

        # 3D Pointed Spread Shirt Collar Wings folding from collar stand over the tie knot sides down to collar points
        collar_inner_edge = 15.0 + 0.88 * np.maximum(0.0, yy - 176.0)
        collar_bot_y = np.where(
            dx_abs <= 78.0,
            176.0 + (238.0 - 176.0) * np.clip((dx_abs - 15.0) / 63.0, 0.0, 1.0),
            238.0 - 1.25 * (dx_abs - 78.0),
        )
        collar_wing_alpha = (
            np.clip((collar_bot_y - yy) / 2.8, 0.0, 1.0)
            * np.clip((dx_abs - collar_inner_edge) / 2.8, 0.0, 1.0)
            * shirt_w
        )
        collar_roll_luma = (1.05 + 0.04 * np.exp(-0.5 * ((yy - (top_contour_y + 16.0)) / 14.0) ** 2)) * cotton_poplin
        under_collar_shadow = (
            np.exp(-0.5 * ((yy - (collar_bot_y + 5.0)) / 6.5) ** 2)
            * ((yy >= collar_bot_y - 1.5) & (dx_abs >= 16.0) & (dx_abs <= v_half_width))
        )
        collar_inner_seam = (
            np.exp(-0.5 * ((dx_abs - collar_inner_edge) / 3.5) ** 2)
            * ((yy >= 176.0) & (yy <= collar_bot_y + 2.0) & (dx_abs <= v_half_width))
        )
        neck_rim_shade = np.exp(-0.5 * ((yy - top_contour_y) / 5.5) ** 2) * (dx_abs <= v_half_width)
        shirt_rgb = shirt_rgb * (1.0 - 0.22 * under_collar_shadow[:, :, None] - 0.14 * collar_inner_seam[:, :, None] - 0.08 * neck_rim_shade[:, :, None])
        wing_rgb = np.clip(c_sec[None, None, :] * (0.95 + 0.05 * nz_torso[:, :, None]) * collar_roll_luma[:, :, None], 0.0, 255.0)
        shirt_rgb = shirt_rgb * (1.0 - collar_wing_alpha[:, :, None]) + wing_rgb * collar_wing_alpha[:, :, None]
        shirt_rgb = np.clip(shirt_rgb, 0.0, 255.0)

        if style == "male_suit":
            # 3D Conical Windsor Silk Tie Knot starting right at the collar stand (y in [174, 242]) & Blade below
            knot_y = (yy >= 174.0) & (yy <= 242.0)
            knot_half_w = 24.0 - 0.10 * (yy - 174.0)
            blade_y = (yy > 242.0) & (yy < canvas_h - 40.0)
            blade_half_w = 17.2 + 0.062 * (yy - 242.0)
            tie_half_w = np.where(knot_y, knot_half_w, np.where(blade_y, blade_half_w, 0.0))
            # Collar wings fold ON TOP of the upper outer corners of the tie knot
            tie_w = np.clip((tie_half_w - dx_abs) / 2.4, 0.0, 1.0) * shirt_w * (1.0 - 0.92 * collar_wing_alpha)

            # 3D Cylindrical Normal Shading across Tie Knot + Diagonal Repp Silk Jacquard Weave + Windsor Dimple
            nx_tie = np.clip((xx - cx) / np.maximum(6.0, tie_half_w), -1.0, 1.0)
            tie_cyl = 0.78 + 0.30 * np.sqrt(np.maximum(0.08, 1.0 - nx_tie * nx_tie))
            dimple = np.exp(-0.5 * (dx_abs / 5.5) ** 2) * np.exp(-0.5 * ((yy - 258.0) / 18.0) ** 2)
            knot_seam = np.exp(-0.5 * ((yy - 242.0) / 4.2) ** 2) * (dx_abs <= tie_half_w)
            top_knot_ao = np.exp(-0.5 * ((yy - top_contour_y) / 5.0) ** 2)
            silk_weave = 0.96 + 0.045 * np.sin((xx + yy) * 0.85) + 0.02 * np.cos((xx - yy) * 1.75)
            tie_shade = tie_cyl * (1.0 - 0.22 * dimple - 0.25 * knot_seam - 0.12 * top_knot_ao) * silk_weave
            tie_rgb = np.clip(c_bord[None, None, :] * tie_shade[:, :, None], 0.0, 255.0)

            # Soft drop shadow cast by tie onto white dress shirt
            tie_drop_shadow = np.exp(-0.5 * ((dx_abs - tie_half_w - 4.0) / 7.0) ** 2) * (dx_abs > tie_half_w) * (yy >= 176.0)
            shirt_rgb *= (1.0 - 0.18 * tie_drop_shadow[:, :, None])
            shirt_rgb = shirt_rgb * (1.0 - tie_w[:, :, None]) + tie_rgb * tie_w[:, :, None]

        # Tailored Notched Suit Lapels with Smooth 3D Studio Roll Highlight, Pick-Stitching & Ambient-Occlusion Seam
        lapel_dist = dx_abs - v_half_width
        lapel_width = 52.0 + 0.030 * np.maximum(0.0, yy - 144.0)
        notch_indent = 4.5 * np.exp(-0.5 * ((yy - 226.0) / 7.5) ** 2)
        eff_lapel_w = lapel_width - notch_indent
        in_lapel = (lapel_dist >= 0.0) & (lapel_dist <= eff_lapel_w) & (yy < canvas_h - 40.0)
        lapel_roll = np.exp(-0.5 * ((lapel_dist - 19.0) / 15.0) ** 2) * in_lapel
        lapel_seam = np.exp(-0.5 * ((lapel_dist - eff_lapel_w) / 5.5) ** 2) * (yy < canvas_h - 40.0)
        notch_crease = np.exp(-0.5 * ((yy - 226.0) / 4.0) ** 2) * ((lapel_dist >= 22.0) & (lapel_dist <= lapel_width))
        lapel_pick_stitch = (
            np.exp(-0.5 * ((lapel_dist - (eff_lapel_w - 4.5)) / 1.4) ** 2)
            * (np.sin(yy * 0.95) > 0.25)
            * (yy < canvas_h - 50.0)
        )

        # Soft cast shadow from suit jacket lapel edge onto the white shirt
        lapel_shirt_shadow = np.exp(-0.5 * ((v_half_width - dx_abs) / 8.5) ** 2) * (dx_abs <= v_half_width)
        shirt_rgb *= (1.0 - 0.20 * lapel_shirt_shadow[:, :, None])

        suit_rgb = suit_rgb * (
            1.0
            + 0.14 * lapel_roll[:, :, None]
            - 0.13 * lapel_seam[:, :, None]
            - 0.06 * notch_crease[:, :, None]
            - 0.04 * lapel_pick_stitch[:, :, None]
        )
        rgb = suit_rgb * (1.0 - shirt_w[:, :, None]) + shirt_rgb * shirt_w[:, :, None]

    garment_rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    garment_alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    _TEMPLATE_CACHE[cache_key] = (garment_rgb, garment_alpha, src_pts)
    return garment_rgb, garment_alpha, src_pts


def load_or_create_studio_garment_asset(
    preset_key: str,
    canvas_w: int = 900,
    canvas_h: int = 720,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or synthesize a high-resolution (900x720 RGBA PNG) studio garment asset on disk in
    `shop_data/garments/<preset_key>.png` along with its 9 canonical TPS landmarks.
    Ensures photorealistic studio garment PNG templates exist on disk for both TPS
    warping and IDM-VTON / CatVTON reference conditioning.
    """
    garment_rgb, garment_alpha, src_pts = build_canonical_garment_template(
        preset_key, canvas_w=canvas_w, canvas_h=canvas_h
    )
    if canvas_w == 900 and canvas_h in (700, 720) and preset_key in DRESS_PRESETS and preset_key != "none":
        try:
            os.makedirs(GARMENTS_DIR, exist_ok=True)
            png_path = os.path.join(GARMENTS_DIR, f"{preset_key}.png")
            need_write = not os.path.isfile(png_path)
            if not need_write and canvas_h == 720:
                try:
                    with Image.open(png_path) as existing_im:
                        if existing_im.size != (canvas_w, canvas_h):
                            need_write = True
                except Exception:
                    need_write = True
            if need_write:
                rgba = np.dstack([
                    garment_rgb,
                    np.clip(np.round(garment_alpha * 255.0), 0, 255).astype(np.uint8),
                ])
                Image.fromarray(rgba, mode="RGBA").save(png_path, format="PNG")
        except Exception:
            pass
    return garment_rgb, garment_alpha, src_pts


# ============================================================================
# 3. CERVICAL SKIN CONTINUITY & FEMALE HAIR OCCLUSION PRESERVATION
# ============================================================================

def _synthesize_cervical_skin_patch(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    dy_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a physically shaded 3D Cylindrical Cervical-Clavicular Skin Field
    matching the subject's exact cheek/jaw CIE L*a*b* chromophores, with natural
    submental chin shadow and beard/jawline preservation so open/scoop/suit necklines
    never show old shirt collars inside the neck opening!
    """
    h, w = img_rgb.shape[:2]
    med_L, med_a, med_b = geom.skin_lab_median
    ipd = max(16.0, geom.ipd_px)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    neck_top_cx = 0.5 * (geom.neck_top_left[0] + geom.neck_top_right[0])
    neck_base_cx = 0.5 * (geom.neck_base_left[0] + geom.neck_base_right[0])
    chin_y = geom.chin_xy[1]
    jug_y = geom.jugular_notch[1] + dy_shift
    neck_half_w = max(12.0, 0.54 * abs(geom.neck_base_right[0] - geom.neck_base_left[0]))

    # Cervical cone alpha mask starting just below the chin apex down to below jugular notch
    y_top = chin_y - 0.04 * ipd
    y_bot = max(y_top + 12.0, jug_y + 0.45 * ipd)
    t_y = np.clip((yy - y_top) / max(8.0, y_bot - y_top), 0.0, 1.0)
    neck_cx_y = neck_top_cx * (1.0 - t_y) + neck_base_cx * t_y

    # 3D Cylindrical neck shading + lateral illumination match + gentle submental shadow
    nx = np.clip((xx - neck_cx_y) / (neck_half_w * 1.15), -1.0, 1.0)
    cyl_lum = np.sqrt(np.maximum(0.28, 1.0 - 0.34 * nx * nx))
    lat_grad = 1.0 - 0.007 * float(np.clip(geom.lateral_shadow_diff, -12.0, 12.0)) * nx
    chin_shadow = 1.0 - 0.14 * np.exp(-0.5 * ((yy - chin_y) / max(4.0, 0.14 * ipd)) ** 2)

    L_neck = np.clip(med_L * cyl_lum * lat_grad * chin_shadow, 18.0, 92.0)
    a_neck = np.full_like(L_neck, med_a)
    b_neck = np.full_like(L_neck, med_b)
    neck_rgb = lab_to_rgb(np.stack([L_neck, a_neck, b_neck], axis=-1))

    cone_half_w = neck_half_w * (0.86 + 0.30 * t_y)

    in_x = np.clip((cone_half_w - np.abs(xx - neck_cx_y)) / max(3.5, 0.07 * ipd), 0.0, 1.0)
    in_y = np.clip((yy - y_top) / max(3.5, 0.07 * ipd), 0.0, 1.0) * np.clip(
        (y_bot - yy) / max(4.0, 0.12 * ipd), 0.0, 1.0
    )

    # Distinguish real neck skin & dark beard/submental shadow from foreign old shirt cloth inside the neckline:
    # Real neck skin (including cool submental shadow) lies near (med_a, med_b) with L* in [22, 83],
    # and dark chin beard has L* < min(48, med_L * 0.76) with low chroma.
    lab_orig = rgb_to_lab(img_rgb)
    L_orig = lab_orig[:, :, 0]
    a_orig = lab_orig[:, :, 1]
    b_orig = lab_orig[:, :, 2]

    chroma_dist = np.hypot(a_orig - med_a, b_orig - med_b)
    # Shadowed neck skin can be slightly desaturated toward neutral (0.45*med_a, 0.45*med_b)
    shadow_chroma_dist = np.hypot(a_orig - 0.50 * med_a, b_orig - 0.50 * med_b)
    effective_chroma_dist = np.minimum(chroma_dist, shadow_chroma_dist)

    is_bright_white_cloth = np.clip((L_orig - max(80.0, med_L + 15.0)) / 8.0, 0.0, 1.0)
    is_dyed_cloth = np.clip((effective_chroma_dist - 9.5) / 5.5, 0.0, 1.0)
    is_luma_mismatch_cloth = np.clip((np.abs(L_orig - L_neck) - 15.0) / 8.0, 0.0, 1.0) * np.clip((0.58 - geom.skin_mask) / 0.30, 0.0, 1.0)
    is_lateral_collar_tab = (
        np.clip((0.55 - geom.skin_mask) / 0.28, 0.0, 1.0)
        * np.clip((np.abs(xx - neck_cx_y) - 0.34 * neck_half_w) / max(3.0, 0.14 * neck_half_w), 0.0, 1.0)
    )
    is_foreign_cloth = np.maximum(
        np.maximum(is_bright_white_cloth, is_dyed_cloth) * (1.0 - 0.85 * geom.skin_mask),
        np.maximum(is_luma_mismatch_cloth, is_lateral_collar_tab),
    )

    # Strictly protect dark chin/submental beard near the chin & upper neck (yy <= chin_y + 0.14*ipd)
    in_chin_beard_zone = np.clip((chin_y + 0.14 * ipd - yy) / max(2.5, 0.05 * ipd), 0.0, 1.0) * np.exp(
        -0.5 * ((xx - neck_top_cx) / max(4.0, 0.56 * ipd)) ** 2
    )
    dark_beard_thresh = min(48.0, med_L * 0.75)
    beard_chin_guard = np.clip((dark_beard_thresh - L_orig) / 10.0, 0.0, 1.0) * in_chin_beard_zone

    neck_patch_alpha = in_x * in_y * is_foreign_cloth * (1.0 - 0.99 * beard_chin_guard)
    neck_patch_alpha = gaussian_blur(neck_patch_alpha, sigma=max(1.5, 0.03 * ipd))
    return neck_rgb, np.clip(neck_patch_alpha, 0.0, 1.0)


def _extract_female_hair_occlusion_matte(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    fg_alpha: np.ndarray | None = None,
) -> np.ndarray:
    """
    Detect long female hair strands falling lateral to the neck over the shoulders/clavicles
    so they are preserved ON TOP of the newly draped Saree/Kurti/Blazer!
    Supports dark, brown, blonde, henna/auburn, and silver/gray hair by sampling the
    subject's cranial hair chromophores and verifying vertical continuity past the jaw gate.
    """
    h, w = img_rgb.shape[:2]
    hm_key = (
        h,
        w,
        int(img_rgb[::max(1, h // 12), ::max(1, w // 12)].sum()),
        round(float(geom.ipd_px), 2),
        fg_alpha is not None,
    )
    if _HAIR_MATTE_CACHE.get("key") == hm_key and _HAIR_MATTE_CACHE.get("matte") is not None:
        return _HAIR_MATTE_CACHE["matte"]  # type: ignore[return-value]

    lab = rgb_to_lab(img_rgb)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    med_L, med_a, med_b = geom.skin_lab_median
    ipd = max(16.0, geom.ipd_px)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    neck_cx = 0.5 * (geom.neck_top_left[0] + geom.neck_top_right[0])
    eye_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)
    crown_y = geom.crown_xy[1]
    chin_y = geom.chin_xy[1]
    jug_y = geom.jugular_notch[1]

    # Estimate background color from top corners to exclude background pixels from cranial hair model
    pad = max(4, int(min(h, w) * 0.05))
    bg_corners = np.concatenate([lab[:pad, :pad].reshape(-1, 3), lab[:pad, w - pad:].reshape(-1, 3)], axis=0)
    bg_med = np.median(bg_corners, axis=0)

    # 1. Sample subject's cranial/temple hair zone above eye-line to learn hair (L*, a*, b*)
    in_cranial_zone = (
        (yy >= max(0.0, crown_y - 0.15 * ipd))
        & (yy <= eye_y + 0.15 * ipd)
        & (np.abs(xx - neck_cx) <= 1.25 * ipd)
        & (geom.skin_mask < 0.55)
    )
    dist_from_bg = np.sqrt(
        0.5 * (L - bg_med[0]) ** 2 + (a - bg_med[1]) ** 2 + (b - bg_med[2]) ** 2
    )
    cranial_candidates = in_cranial_zone & (dist_from_bg > 12.0)
    if fg_alpha is not None:
        af_check = fg_alpha.astype(np.float32)
        if af_check.max() > 1.5:
            af_check /= 255.0
        cranial_candidates = cranial_candidates & (af_check > 0.5)

    # Classic dark-hair probability + adaptive cranial-chromophore similarity
    dark_thresh = min(38.0, med_L * 0.56)
    dark_prob = np.clip((dark_thresh - L) / 14.0, 0.0, 1.0)

    if np.count_nonzero(cranial_candidates) >= 20:
        hair_pixels = lab[cranial_candidates]
        h_med = np.median(hair_pixels, axis=0)
        h_std = np.maximum(np.std(hair_pixels, axis=0), np.array([10.0, 6.0, 6.0], dtype=np.float32))
        d_hair = np.sqrt(
            0.45 * ((L - h_med[0]) / h_std[0]) ** 2
            + 0.85 * ((a - h_med[1]) / h_std[1]) ** 2
            + 0.85 * ((b - h_med[2]) / h_std[2]) ** 2
        )
        # Ensure non-background & distinct from facial skin median
        bg_gate = np.clip((dist_from_bg - 10.0) / 10.0, 0.0, 1.0)
        chroma_hair_prob = np.exp(-0.45 * (d_hair ** 2)) * bg_gate
        hair_color_prob = np.maximum(dark_prob, chroma_hair_prob)
    else:
        hair_color_prob = dark_prob

    not_skin_gate = np.clip((0.68 - geom.skin_mask) / 0.45, 0.0, 1.0)
    non_skin_hair = hair_color_prob * not_skin_gate

    # 2. Lateral Jaw-Chin Continuity Gate:
    # Real long hair draping over the shoulders must pass through the sub-auricular lateral jaw gate
    # (yy in [chin_y - 0.14*ipd, chin_y - 0.01*ipd], lateral_dist in [0.72*ipd, 1.35*ipd]).
    # This prevents a short-haired woman's dark old shirt on her shoulders from being mistaken for hair!
    dx = xx - neck_cx
    lateral_dist = np.abs(dx)
    jaw_gate_band = (
        (yy >= chin_y - 0.14 * ipd)
        & (yy <= chin_y - 0.01 * ipd)
        & (lateral_dist >= 0.72 * ipd)
        & (lateral_dist <= 1.35 * ipd)
    )
    left_jaw_mask = jaw_gate_band & (dx < 0.0)
    right_jaw_mask = jaw_gate_band & (dx >= 0.0)

    left_jaw_density = float(np.mean(non_skin_hair[left_jaw_mask])) if np.any(left_jaw_mask) else 0.0
    right_jaw_density = float(np.mean(non_skin_hair[right_jaw_mask])) if np.any(right_jaw_mask) else 0.0

    left_continuity = float(np.clip((left_jaw_density - 0.12) / 0.18, 0.0, 1.0))
    right_continuity = float(np.clip((right_jaw_density - 0.12) / 0.18, 0.0, 1.0))
    side_continuity = np.where(dx < 0.0, left_continuity, right_continuity)

    # Hair strands falling over shoulders sit laterally outside the central neck core
    in_hair_corridor = (
        (lateral_dist > 0.42 * ipd)
        & (lateral_dist < 1.45 * ipd)
        & (yy > eye_y + 0.65 * ipd)
        & (yy < jug_y + 0.55 * ipd)
    )

    raw_hair = non_skin_hair * side_continuity * in_hair_corridor.astype(np.float32)

    if fg_alpha is not None:
        af = fg_alpha.astype(np.float32)
        if af.max() > 1.5:
            af /= 255.0
        raw_hair *= af

    radius = max(3, int(ipd * 0.05))
    hair_matte = guided_filter(img_rgb, raw_hair, radius=radius, eps=8e-4)
    out_matte = np.clip(hair_matte, 0.0, 0.92).astype(np.float32)
    _HAIR_MATTE_CACHE["key"] = hm_key
    _HAIR_MATTE_CACHE["matte"] = out_matte
    return out_matte


# ============================================================================
# 4. HYBRID REAL DIFFUSION AI VIRTUAL TRY-ON / INPAINTING ENGINE
# ============================================================================

_DIFFUSION_INPAINT_PIPE = None
_DIFFUSION_INPAINT_TRIED = False

DIFFUSION_DRESS_PROMPTS: dict[str, str] = {
    "female_saree_maroon_gold": "photorealistic studio passport portrait wearing a traditional rich maroon Banarasi silk saree with 24k gold zari border and tailored maroon blouse, studio lighting, 8k fabric detail",
    "female_saree_royal_blue": "photorealistic studio passport portrait wearing a royal blue Kanjeevaram silk saree with gold zari border and matching silk blouse, studio lighting, 8k fabric detail",
    "female_salwar_emerald": "photorealistic studio passport portrait wearing an emerald green formal salwar kameez suit with crimson gold embroidered dupatta, studio lighting, 8k fabric detail",
    "female_kurti_ivory_gold": "photorealistic studio passport portrait wearing a classic warm ivory silk kurti with subtle gold chikankari embroidery collar, studio lighting, 8k fabric detail",
    "female_blazer_navy": "photorealistic studio passport portrait wearing a tailored corporate navy blue women blazer over a crisp white silk inner blouse, studio lighting, 8k wool fabric texture",
    "female_blazer_charcoal": "photorealistic studio passport portrait wearing a tailored charcoal grey formal suit jacket over a soft cream silk blouse, studio lighting, 8k wool fabric texture",
    "female_shirt_sky_blue": "photorealistic studio passport portrait wearing a crisp sky blue formal collared office shirt, studio lighting, 8k cotton weave detail",
    "male_suit_navy_tie": "photorealistic studio passport portrait wearing a bespoke navy blue executive business suit jacket, crisp white spread-collar dress shirt, and burgundy red silk Windsor tie, studio lighting, 8k wool weave",
    "male_suit_charcoal_tie": "photorealistic studio passport portrait wearing a bespoke charcoal black formal business suit jacket, crisp white dress shirt, and royal blue silk Windsor tie, studio lighting, 8k wool weave",
    "male_blazer_royal": "photorealistic studio passport portrait wearing a tailored navy blue blazer over a crisp white open-collar dress shirt, no tie, studio lighting, 8k fabric detail",
    "male_shirt_white": "photorealistic studio passport portrait wearing a crisp studio white formal collared dress shirt, studio lighting, 8k cotton weave detail",
    "male_shirt_sky_blue": "photorealistic studio passport portrait wearing a crisp sky blue formal collared dress shirt, studio lighting, 8k cotton weave detail",
}


def build_diffusion_clothing_inpaint_mask(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    vertical_shift: float = 0.0,
    preset_key: str = "male_suit_navy_tie",
) -> tuple[np.ndarray, str, str]:
    """
    Generate an anthropometric clothing inpainting mask (uint8 H x W in [0, 255])
    that covers 100% of the torso/clothing region below the chin and mandible while
    strictly protecting (mask = 0) the face, eyes, eyeglasses, beard, chin, and upper neck core.
    Returns (mask_u8, base_prompt, negative_prompt).
    """
    h, w = img_rgb.shape[:2]
    ipd = max(16.0, geom.ipd_px)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    eye_mid_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)
    face_cx = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    neck_top_cx = 0.5 * (geom.neck_top_left[0] + geom.neck_top_right[0])
    neck_base_cx = 0.5 * (geom.neck_base_left[0] + geom.neck_base_right[0])
    skeletal_chin_y = min(geom.chin_xy[1], eye_mid_y + 1.58 * ipd) + min(0.0, vertical_shift * ipd * 0.20)

    # Parabolic sub-mandibular boundary below chin/jawline (clamped so it never rises into upper face at image borders)
    jaw_curve = np.minimum(1.2, ((xx - face_cx) / max(12.0, 0.68 * ipd)) ** 2)
    jaw_protect_y = skeletal_chin_y - 0.12 * ipd * jaw_curve
    below_jaw = np.clip((yy - jaw_protect_y) / max(3.0, 0.05 * ipd), 0.0, 1.0)

    # Protect upper central neck skin above the tie knot / collar button
    jug_y = geom.jugular_notch[1] + vertical_shift * ipd * 0.25
    t_ny = np.clip((yy - skeletal_chin_y) / max(6.0, jug_y - skeletal_chin_y), 0.0, 1.0)
    neck_cx_y = neck_top_cx * (1.0 - t_ny) + neck_base_cx * t_ny
    upper_neck_core = (
        np.clip((jug_y - 0.12 * ipd - yy) / max(3.5, 0.08 * ipd), 0.0, 1.0)
        * np.clip((0.44 * ipd - np.abs(xx - neck_cx_y)) / max(3.0, 0.05 * ipd), 0.0, 1.0)
    )
    inpaint_float = below_jaw * (1.0 - upper_neck_core)
    inpaint_float = gaussian_blur(inpaint_float, sigma=2.0)
    mask_u8 = np.clip(np.round(inpaint_float * 255.0), 0, 255).astype(np.uint8)

    base_prompt = DIFFUSION_DRESS_PROMPTS.get(
        preset_key,
        DIFFUSION_DRESS_PROMPTS.get(
            "male_suit_navy_tie",
            "photorealistic studio passport portrait in formal attire, sharp studio lighting, 8k",
        ),
    )
    neg_prompt = (
        "deformed collar, wide white collar wings, crooked tie, extra limbs, asymmetric lapels, "
        "blurry fabric, cartoon, illustration, low quality, watermark, exposed old shirt"
    )
    return mask_u8, base_prompt, neg_prompt


def try_diffusion_dress_inpaint(
    init_dressed_rgb: np.ndarray,
    preset_key: str,
    geom: FacialAnthropometrics,
    vertical_shift: float = 0.0,
) -> np.ndarray | None:
    """
    Execute V6 True AI Dress Change (ZeroGPU/CUDA `diffusers.AutoPipelineForInpainting`,
    optional `IDM-VTON` / `CatVTON` Gradio Space client, or Hugging Face Serverless `InferenceClient`)
    enabled by default (`PASSPORT_ENABLE_DIFFUSION_DRESS="1"`).
    Crops the lower-jaw + shoulder/torso ROI box (`y_top = max(0, int(chin_y - 0.25 * ipd))` to `h`)
    so 100% of diffusion resolution goes to the suit/saree/collar/shoulders, and composites
    strictly inside `mask_u8` below the jawline while 100% locking the customer's face, eyes, and hair.
    Automatically skips remote network calls during offline unit tests (`PASSPORT_OFFLINE_TESTS="1"` or `unittest`).
    """
    global _DIFFUSION_INPAINT_PIPE, _DIFFUSION_INPAINT_TRIED

    if os.environ.get("PASSPORT_ENABLE_DIFFUSION_DRESS", "0").strip() not in ("1", "true", "TRUE", "yes"):
        return None
    if os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return None
    if "unittest" in sys.modules and os.environ.get("PASSPORT_FORCE_DIFFUSION_TEST", "0") != "1":
        return None

    h, w = init_dressed_rgb.shape[:2]
    ipd = max(16.0, geom.ipd_px)
    mask_u8, prompt, neg_prompt = build_diffusion_clothing_inpaint_mask(
        init_dressed_rgb, geom=geom, vertical_shift=vertical_shift, preset_key=preset_key
    )

    # Crop lower-jaw + shoulder/torso ROI box so 100% of diffusion resolution goes to the garment
    y_top = max(0, min(h - 32, int(round(geom.chin_xy[1] - 0.25 * ipd))))
    roi_rgb = init_dressed_rgb[y_top:h, 0:w]
    roi_mask = mask_u8[y_top:h, 0:w]
    rh, rw = roi_rgb.shape[:2]
    if rh < 24 or rw < 24:
        return None

    scale_d = min(1.0, 768.0 / float(max(rh, rw)))
    dw = max(64, int(round((rw * scale_d) / 8.0)) * 8)
    dh = max(64, int(round((rh * scale_d) / 8.0)) * 8)

    def _composite_roi(gen_pil: Image.Image) -> np.ndarray:
        diff_roi = np.array(gen_pil.resize((rw, rh), Image.Resampling.LANCZOS), dtype=np.float32)
        alpha_m = (roi_mask.astype(np.float32) / 255.0)[:, :, None]
        out_full = init_dressed_rgb.astype(np.float32)
        out_full[y_top:h, 0:w] = roi_rgb.astype(np.float32) * (1.0 - alpha_m) + diff_roi * alpha_m
        return np.clip(np.round(out_full), 0, 255).astype(np.uint8)

    # Path A: Local ZeroGPU / CUDA `diffusers.AutoPipelineForInpainting`
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            from diffusers import AutoPipelineForInpainting  # type: ignore

            if _DIFFUSION_INPAINT_PIPE is None and not _DIFFUSION_INPAINT_TRIED:
                _DIFFUSION_INPAINT_TRIED = True
                model_id = os.environ.get(
                    "PASSPORT_DIFFUSION_INPAINT_MODEL",
                    "runwayml/stable-diffusion-inpainting",
                )
                try:
                    _DIFFUSION_INPAINT_PIPE = AutoPipelineForInpainting.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                    ).to("cuda")
                    _DIFFUSION_INPAINT_PIPE.set_progress_bar_config(disable=True)
                except Exception:
                    _DIFFUSION_INPAINT_PIPE = None

            if _DIFFUSION_INPAINT_PIPE is not None:
                pil_init = Image.fromarray(roi_rgb).resize((dw, dh), Image.Resampling.LANCZOS)
                pil_mask = Image.fromarray(roi_mask).resize((dw, dh), Image.Resampling.BILINEAR)
                out_pil = _DIFFUSION_INPAINT_PIPE(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image=pil_init,
                    mask_image=pil_mask,
                    num_inference_steps=18,
                    strength=0.35,
                    guidance_scale=6.5,
                ).images[0]
                return _composite_roi(out_pil)
    except Exception:
        pass

    # Path B: Gradio Client Virtual Try-On Space (`yisol/IDM-VTON` or `zhengchong/CatVTON`) when configured
    vton_space = os.environ.get("PASSPORT_VTON_SPACE", "").strip()
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or ""
    ).strip()
    if vton_space:
        try:
            import tempfile
            from gradio_client import Client, handle_file  # type: ignore

            garment_png = os.path.join(GARMENTS_DIR, f"{preset_key}.png")
            if not os.path.isfile(garment_png):
                load_or_create_studio_garment_asset(preset_key, canvas_w=900, canvas_h=720)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf_p:
                person_tmp = tf_p.name
                Image.fromarray(roi_rgb).resize((dw, dh), Image.Resampling.LANCZOS).save(person_tmp, format="PNG")
            try:
                vton_client = Client(vton_space, hf_token=hf_token or None)
                res_vton = vton_client.predict(
                    dict={"background": handle_file(person_tmp), "layers": [], "composite": None},
                    garm_img=handle_file(garment_png),
                    garment_des=prompt,
                    is_checked=True,
                    is_checked_crop=False,
                    denoise_steps=20,
                    seed=42,
                    api_name="/tryon",
                )
                out_path = res_vton[0] if isinstance(res_vton, (list, tuple)) else res_vton
                if isinstance(out_path, str) and os.path.isfile(out_path):
                    with Image.open(out_path) as gen_im:
                        return _composite_roi(gen_im.convert("RGB"))
            finally:
                try:
                    os.remove(person_tmp)
                except OSError:
                    pass
        except Exception:
            pass

    # Path C: Hugging Face Serverless Inference API (`InferenceClient.image_to_image` / inpainting)
    if hf_token:
        try:
            import io
            from huggingface_hub import InferenceClient  # type: ignore

            model_id = os.environ.get(
                "PASSPORT_DIFFUSION_INPAINT_MODEL",
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            )
            client = InferenceClient(token=hf_token, timeout=14.0)
            buf_img = io.BytesIO()
            Image.fromarray(roi_rgb).resize((dw, dh), Image.Resampling.LANCZOS).save(buf_img, format="PNG")
            out_pil = client.image_to_image(
                image=buf_img.getvalue(),
                prompt=prompt,
                negative_prompt=neg_prompt,
                model=model_id,
                strength=0.32,
                guidance_scale=6.5,
            )
            return _composite_roi(out_pil)
        except Exception:
            return None

    return None


def _harmonize_garment_lighting_laplacian(
    warped_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    xx: np.ndarray,
    yy: np.ndarray,
    neck_cx_y: np.ndarray,
) -> np.ndarray:
    """
    2D Multi-Band Laplacian Studio Illumination, Exposure & Color-Temperature Harmonizer:
    Transfers the portrait's directional lateral shadow balance, vertical studio falloff,
    facial exposure level, and subtle ambient white-balance warmth onto the warped garment
    so studio garment compositing achieves seamless photographic integration.
    """
    ipd = max(16.0, geom.ipd_px)
    lat_diff = float(np.clip(geom.lateral_shadow_diff, -12.0, 12.0))
    nx_garment = np.clip((xx - neck_cx_y) / max(24.0, 1.8 * ipd), -1.0, 1.0)
    ny_garment = np.clip((yy - geom.jugular_notch[1]) / max(24.0, 2.2 * ipd), -0.5, 1.0)

    # Match garment exposure subtly to subject's facial luminance (med_L)
    med_L, med_a, med_b = geom.skin_lab_median
    exposure_scale = float(np.clip(0.92 + 0.08 * (med_L / 68.0), 0.88, 1.06))

    # Low-frequency dipole + vertical studio irradiance field
    illum_mod = exposure_scale * (1.0 - (0.0065 * lat_diff) * nx_garment) * (1.02 - 0.04 * np.maximum(0.0, ny_garment))
    out = warped_rgb * illum_mod[:, :, None]

    # Subtle ambient color-temperature harmonization from facial skin median chromophores
    warmth_shift = float(np.clip((med_b - 18.0) * 0.18, -2.5, 2.5))
    if abs(warmth_shift) > 0.2:
        out[:, :, 0] = out[:, :, 0] + warmth_shift * 0.65
        out[:, :, 2] = out[:, :, 2] - warmth_shift * 0.65

    return np.clip(out, 0.0, 255.0)


# ============================================================================
# 5. MASTER SARTORIALFIT™ V6 HYBRID DRESS APPLICATION PIPELINE
# ============================================================================

def _apply_single_subject_sartorial_fallback(
    img_rgb: np.ndarray,
    preset_key: str,
    geom: FacialAnthropometrics | None = None,
    fg_alpha: np.ndarray | None = None,
    vertical_shift: float = 0.0,
    shoulder_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Zero-Download Local Analytical Fallback (`0 MB` on Local Windows PC) for SartorialFit™
    when not running on Hugging Face Cloud (`not IS_HF_CLOUD`).
    """
    if not preset_key or preset_key == "none" or preset_key not in DRESS_PRESETS:
        return img_rgb.copy(), fg_alpha

    h, w = img_rgb.shape[:2]
    if geom is None or (geom.img_w, geom.img_h) != (w, h):
        geom = analyze_face_geometry(img_rgb, fg_alpha)

    from src.core.neural_hub import parse_portrait_semantics
    sem = parse_portrait_semantics(img_rgb, geom=geom)
    has_bisenet = float(sem["bisenet_used"][0]) > 0.5

    preset = DRESS_PRESETS[preset_key]
    is_female = preset.get("gender") == "female"
    style = preset.get("style", "")

    # Load or create 4K-ready 900x720 studio garment PNG asset in shop_data/garments/<preset_key>.png
    tmpl_rgb, tmpl_alpha, src_pts = load_or_create_studio_garment_asset(preset_key, canvas_w=900, canvas_h=720)

    # Compute target 9 anthropometric landmarks on the user's portrait
    ipd = geom.ipd_px
    collar_snug_lift = -0.115 * ipd if style in ("male_suit", "male_blazer", "male_shirt", "female_blazer", "female_shirt") else -0.045 * ipd
    dy_shift = vertical_shift * ipd * 0.25 + collar_snug_lift
    neck_top_cx = 0.5 * (geom.neck_top_left[0] + geom.neck_top_right[0])
    neck_base_cx = 0.5 * (geom.neck_base_left[0] + geom.neck_base_right[0])

    def _scale_top_x(pt: tuple[float, float], s: float) -> list[float]:
        return [neck_top_cx + (pt[0] - neck_top_cx) * s, pt[1] + dy_shift]

    def _scale_base_x(pt: tuple[float, float], s: float) -> list[float]:
        return [neck_base_cx + (pt[0] - neck_base_cx) * s, pt[1] + dy_shift]

    bot_y = float(max(h + int(0.08 * h), int(geom.jugular_notch[1] + 2.45 * ipd)))
    sh_left = _scale_base_x(geom.shoulder_left, shoulder_scale)
    sh_right = _scale_base_x(geom.shoulder_right, shoulder_scale)

    dst_pts = np.array([
        _scale_top_x(geom.neck_top_left, 1.03),
        _scale_top_x(geom.neck_top_right, 1.03),
        _scale_base_x(geom.neck_base_left, 1.02),
        _scale_base_x(geom.neck_base_right, 1.02),
        [geom.jugular_notch[0], geom.jugular_notch[1] + dy_shift],
        sh_left,
        sh_right,
        [sh_left[0] - 0.22 * ipd, bot_y],
        [sh_right[0] + 0.22 * ipd, bot_y],
    ], dtype=np.float32)

    tmpl_rgba = np.dstack([tmpl_rgb, np.clip(np.round(tmpl_alpha * 255.0), 0, 255).astype(np.uint8)])
    warped_rgba = warp_image_tps(tmpl_rgba, src_pts, dst_pts, out_shape=(h, w), grid_step=8, reg=2e-3)

    warped_rgb = warped_rgba[:, :, :3].astype(np.float32)
    warped_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0

    # Strictly prevent any garment pixel from encroaching onto the chin/jaw/face!
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    eye_mid_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)
    center_chin_y = geom.chin_xy[1] + min(0.0, vertical_shift * ipd * 0.20)
    face_cx = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    lat_u2 = np.minimum(1.0, ((xx - face_cx) / max(12.0, 0.68 * ipd)) ** 2)
    jaw_protect_y = center_chin_y + 0.005 * ipd - 0.055 * ipd * lat_u2
    below_jaw = np.clip((yy - jaw_protect_y) / max(3.0, 0.05 * ipd), 0.0, 1.0)
    warped_alpha *= below_jaw

    warped_alpha = gaussian_blur(warped_alpha, sigma=1.1)

    # Parabolic anatomical sub-jaw collar-wing boundary:
    # - Stays strictly below chin tip (`center_chin_y + 0.02 * ipd`) at the central chin/beard
    # - Curves naturally upward laterally outside the jaw (`|x - face_cx| > 0.52 * ipd`) to erase old shirt collar tips
    #   while never approaching the jawline, cheeks, or earlobes (`>= eye_mid_y + 1.24 * ipd`).
    lat_collar_u = np.clip((np.abs(xx - face_cx) - 0.44 * ipd) / max(6.0, 0.28 * ipd), 0.0, 1.0)
    parabolic_cut_y = center_chin_y + 0.02 * ipd - (0.34 * ipd) * (lat_collar_u ** 1.5)
    neck_collar_cut_y = np.maximum(parabolic_cut_y, eye_mid_y + 1.24 * ipd)
    below_neck_cut = np.clip((yy - neck_collar_cut_y) / max(2.5, 0.045 * ipd), 0.0, 1.0)
    if has_bisenet:
        cloth_above_cut = sem["cloth"] * np.clip((yy - (eye_mid_y + 1.10 * ipd)) / max(3.0, 0.06 * ipd), 0.0, 1.0)
        below_neck_cut = np.clip(np.maximum(below_neck_cut, cloth_above_cut), 0.0, 1.0)

    # Step A: Smooth Cylindrical Neck Contour & Old Collar Wing Eradication strictly below chin/jaw
    neck_top_y = geom.neck_top_left[1] + dy_shift
    neck_base_y = geom.neck_base_left[1] + dy_shift
    neck_top_half_w = max(10.0, 0.50 * abs(geom.neck_top_right[0] - geom.neck_top_left[0]) * shoulder_scale)
    neck_base_half_w = max(neck_top_half_w, 0.49 * abs(geom.neck_base_right[0] - geom.neck_base_left[0]) * shoulder_scale)

    t_neck_y = np.clip((yy - neck_top_y) / max(6.0, neck_base_y - neck_top_y), 0.0, 1.0)
    neck_cx_y = neck_top_cx * (1.0 - t_neck_y) + neck_base_cx * t_neck_y
    neck_col_half_w = neck_top_half_w * (1.0 - t_neck_y) + neck_base_half_w * t_neck_y

    warped_rgb = _harmonize_garment_lighting_laplacian(warped_rgb, geom, xx, yy, neck_cx_y)

    base_img = img_rgb.astype(np.float32)
    neck_skin_rgb, neck_skin_alpha = _synthesize_cervical_skin_patch(img_rgb, geom, dy_shift=dy_shift)

    real_neck_skin_gate = np.clip((geom.skin_mask - 0.35) / 0.30, 0.0, 1.0) * np.clip(1.0 - 1.8 * neck_skin_alpha, 0.0, 1.0)
    eff_neck_half_w = neck_col_half_w * (1.0 + 0.12 * real_neck_skin_gate)
    neck_core_mask = np.clip((eff_neck_half_w - np.abs(xx - neck_cx_y)) / max(2.5, 0.04 * ipd), 0.0, 1.0)
    above_jugular = 1.0 - np.clip((yy - (geom.jugular_notch[1] + dy_shift + 0.10 * ipd)) / max(4.0, 0.12 * ipd), 0.0, 1.0)
    valid_neck_skin_col = neck_core_mask * above_jugular
    if has_bisenet:
        valid_neck_skin_col = np.clip(
            np.maximum(valid_neck_skin_col, sem["neck"] * above_jugular) * (1.0 - 0.85 * sem["cloth"]),
            0.0,
            1.0,
        )
        neck_skin_alpha = np.clip(np.maximum(neck_skin_alpha, sem["cloth"] * valid_neck_skin_col), 0.0, 1.0)

    # Step B: Universal Cervical-Clavicular Skin Continuity inside collar/neckline openings
    bg_hole_fill = np.zeros_like(neck_skin_alpha)
    if fg_alpha is not None:
        af_norm = fg_alpha.astype(np.float32)
        if af_norm.max() > 1.5:
            af_norm /= 255.0
        bg_hole_fill = valid_neck_skin_col * np.clip((0.65 - af_norm) / 0.45, 0.0, 1.0)
    open_neckline_zone = (
        np.clip(np.maximum(neck_skin_alpha * valid_neck_skin_col, bg_hole_fill), 0.0, 1.0)
        * (1.0 - warped_alpha)
        * below_jaw
    )
    base_img = base_img * (1.0 - open_neckline_zone[:, :, None]) + (
        neck_skin_rgb.astype(np.float32) * open_neckline_zone[:, :, None]
    )

    # Step C: Female Long-Hair Occlusion Guard (preserve hair strands falling over shoulders)
    hair_matte = np.zeros((h, w), dtype=np.float32)
    if is_female:
        hair_matte = _extract_female_hair_occlusion_matte(img_rgb, geom, fg_alpha)
        if has_bisenet:
            hair_matte = np.clip(np.maximum(hair_matte, sem["hair"] * below_jaw * 0.90), 0.0, 0.92)
        effective_dress_alpha = warped_alpha * (1.0 - 0.85 * hair_matte)
    else:
        effective_dress_alpha = warped_alpha

    # Replace old collar wings strictly BELOW `neck_collar_cut_y` & outside neck cylinder & outside new garment!
    pad_bg = max(4, int(min(h, w) * 0.05))
    bg_corners = np.concatenate([
        img_rgb[:pad_bg, :pad_bg, :3].reshape(-1, 3),
        img_rgb[:pad_bg, max(0, w - pad_bg):, :3].reshape(-1, 3),
    ], axis=0)
    bg_est_rgb = np.median(bg_corners, axis=0).astype(np.float32)
    strict_exterior_bg = np.clip((0.02 - valid_neck_skin_col) / 0.02, 0.0, 1.0)
    old_collar_wing_zone = below_neck_cut * strict_exterior_bg * (1.0 - warped_alpha) * (1.0 - hair_matte)
    base_img = base_img * (1.0 - old_collar_wing_zone[:, :, None]) + bg_est_rgb[None, None, :] * old_collar_wing_zone[:, :, None]

    # Step D: Subtle Ambient Occlusion Collar Shadow cast by garment onto neck skin
    collar_rim = gaussian_blur(warped_alpha, sigma=max(2.0, 0.035 * ipd)) * (1.0 - warped_alpha) * valid_neck_skin_col
    base_img *= (1.0 - 0.16 * collar_rim[:, :, None])

    # Composite warped garment over portrait
    comp_rgb = base_img * (1.0 - effective_dress_alpha[:, :, None]) + (
        warped_rgb * effective_dress_alpha[:, :, None]
    )
    out_rgb = np.clip(np.round(comp_rgb), 0, 255).astype(np.uint8)

    # Optional Step E: Hybrid Real Diffusion AI Inpainting Refinement (when ZeroGPU/CUDA or HF API diffusion is enabled)
    diff_refined = try_diffusion_dress_inpaint(
        out_rgb,
        preset_key=preset_key,
        geom=geom,
        vertical_shift=vertical_shift,
    )
    if diff_refined is not None:
        out_rgb = diff_refined

    # Update foreground alpha matte: 100% preserve original head/cheeks/ears/jawline above `neck_collar_cut_y`!
    out_alpha = None
    if fg_alpha is not None:
        af = fg_alpha.astype(np.float32)
        scale_255 = af.max() > 1.5
        if scale_255:
            af /= 255.0
        valid_below_neck_af = np.maximum(
            warped_alpha,
            np.maximum(valid_neck_skin_col * af, af * hair_matte),
        )
        new_af = af * (1.0 - below_neck_cut) + valid_below_neck_af * below_neck_cut
        new_af = np.maximum(new_af, warped_alpha)
        out_alpha = np.clip(np.round(new_af * 255.0), 0, 255).astype(np.uint8) if scale_255 else np.clip(new_af, 0.0, 1.0)

    return out_rgb, out_alpha


def apply_anthropometric_dress(
    img_rgb: np.ndarray,
    preset_key: str,
    geom: FacialAnthropometrics | None = None,
    fg_alpha: np.ndarray | None = None,
    vertical_shift: float = 0.0,
    shoulder_scale: float = 1.0,
    target_person: str = "auto",
    custom_garment_rgb: np.ndarray | None = None,
    subjects: list | None = None,
    current_dressed_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Master V8 AI Virtual Try-On & Multi-Subject Official Dress Engine:
    - In Hugging Face Cloud (`IS_HF_CLOUD == True`), executes real FP16 CUDA `CatVTON`
      (`zhengchong/CatVTON` `mix-48k-1024`) with Photorealistic Studio Garment References
      (or user-uploaded `custom_garment_rgb`) and Multi-Subject / Official Couple Routing!
    - On Local Windows PC (`not IS_HF_CLOUD`, `0 MB` local model weights), executes fast
      multi-subject `SartorialFit™` analytical warping per subject `roi_box`.
    """
    has_custom = custom_garment_rgb is not None and isinstance(custom_garment_rgb, np.ndarray) and custom_garment_rgb.size > 0
    if not has_custom and (not preset_key or preset_key == "none" or preset_key not in DRESS_PRESETS):
        return img_rgb.copy(), fg_alpha

    # 1. Primary Cloud Engine: Real FP16 CUDA `CatVTON` Virtual Try-On
    from src.shared.config import IS_HF_CLOUD
    if IS_HF_CLOUD:
        try:
            from src.core.vton_ai_engine import apply_catvton_virtual_tryon

            vton_res = apply_catvton_virtual_tryon(
                clean_base_rgb=img_rgb,
                current_dressed_rgb=current_dressed_rgb,
                clean_fg_alpha=fg_alpha,
                preset_key=preset_key,
                target_person=target_person,
                custom_garment_rgb=custom_garment_rgb,
                subjects=subjects,
                vertical_shift=vertical_shift,
                shoulder_scale=shoulder_scale,
            )
            if vton_res is not None:
                return vton_res[0], vton_res[1]
        except Exception as e:
            print(f"⚠️ [DressEngine V8] CatVTON cloud fallback triggered: {e}")

    # 2. Local Zero-Download Multi-Subject Analytical Fallback (`0 MB` local weights)
    from src.core.multi_subject_detector import detect_foreground_subjects
    from src.core.vton_ai_engine import resolve_subject_assignments

    active_subjects = subjects if subjects else detect_foreground_subjects(img_rgb, raw_bria_alpha=fg_alpha)
    assignments = resolve_subject_assignments(
        active_subjects,
        preset_key=preset_key,
        target_person=target_person,
        has_custom_garment=False,
    )
    if not assignments:
        return img_rgb.copy(), fg_alpha

    working_rgb = (
        current_dressed_rgb.copy()
        if (current_dressed_rgb is not None and current_dressed_rgb.shape == img_rgb.shape)
        else img_rgb.copy()
    )
    working_alpha = fg_alpha.copy() if fg_alpha is not None else None
    for subj, sub_key in assignments:
        rx1, _ry1, rx2, _ry2 = subj.roi_box
        sub_rgb, sub_af = _apply_single_subject_sartorial_fallback(
            img_rgb,
            preset_key=sub_key,
            geom=subj.geom,
            fg_alpha=fg_alpha,
            vertical_shift=vertical_shift,
            shoulder_scale=shoulder_scale,
        )
        working_rgb[:, rx1:rx2] = sub_rgb[:, rx1:rx2]
        if working_alpha is not None and sub_af is not None:
            working_alpha[:, rx1:rx2] = sub_af[:, rx1:rx2]
    return working_rgb, working_alpha


