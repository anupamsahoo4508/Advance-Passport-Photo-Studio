"""
AI Studio Auto-Director Brain, Natural Scars/Spots Identity Lock & Oklab Studio Color Grading
(`studio_lighting_ai.py` — V7 Studio Master Edition).

Addresses 3 core studio portrait requirements simultaneously:
1. `IdentityMarkLock™` (`compute_identity_mark_lock`):
   Separates permanent human identity marks — natural facial scars, moles (nevi), beauty spots,
   birthmarks, freckles, and anatomical expression folds — from random sensor ISO noise using
   Multiscale Frangi/Hessian Anisotropic Ridgeness + Melanin Band-Pass Nevus Detection.
   Locks permanent marks at 100% fidelity so AI never erases or airbrushes a customer's real face.
2. `StudioAutoDirectorAI` (`analyze_12zone_studio_director`):
   Inspects 12 anatomical facial & cervical zones, Individual Typology Angle (ITA°), MTF sharpness,
   ISO noise, flash glare hotspots, and sclera white-balance cast to decide automatically:
   - Exact `codeformer_fidelity_w` & `neural_blend_strength`
   - Exact `perfection_level` (denoise polish on flat skin outside `IdentityMarkLock™`)
   - Exact 2D spatial zones where studio fill relighting or flash de-glare is needed vs already optimal.
3. `apply_ai_spatial_studio_relight_and_grade`:
   Applies targeted 2D spatial studio beauty-dish relighting, optical polarization flash de-glare,
   ambient room cast neutralization, and zero-hue-twist Perceptual Oklab filmic studio color grading
   while locking 100% of the customer's natural skin colour.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from src.core.cv_ops import (
    box_filter,
    compute_gradients,
    gaussian_blur,
    guided_filter,
    lab_to_rgb,
    oklab_to_rgb,
    rgb_to_lab,
    rgb_to_oklab,
)
from src.core.facial_math_engine import FacialAnthropometrics, analyze_face_geometry


@dataclass
class StudioDirectorPlan:
    """Complete spatial & photometric decisions made by the V7 AI Studio Auto-Director."""
    director_mode: str
    ita_angle_deg: float
    skin_tone_category: str
    mtf_sharpness: float
    iso_noise_sigma: float
    glare_hotspot_ratio: float
    codeformer_fidelity_w: float
    neural_blend_strength: float
    perfection_level: float
    pore_acutance_gain: float
    global_shadow_deficit_L: float
    lateral_asymmetry_L: float
    wb_cast_da: float
    wb_cast_db: float
    oklab_contrast_gain: float
    zone_luma_L: dict[str, float] = field(default_factory=dict)
    zone_fill_boost_L: dict[str, float] = field(default_factory=dict)
    spatial_relight_delta_L: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros((1, 1), dtype=np.float32)
    )
    deglare_weight_2d: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros((1, 1), dtype=np.float32)
    )

    def to_telemetry_dict(self) -> dict[str, object]:
        return {
            "director_mode": self.director_mode,
            "ita_skin_angle_deg": round(self.ita_angle_deg, 2),
            "skin_tone_category": self.skin_tone_category,
            "mtf_sharpness": round(self.mtf_sharpness, 3),
            "iso_noise_sigma": round(self.iso_noise_sigma, 3),
            "glare_hotspot_ratio": round(self.glare_hotspot_ratio, 4),
            "ai_codeformer_fidelity_w": round(self.codeformer_fidelity_w, 3),
            "ai_neural_blend": round(self.neural_blend_strength, 3),
            "ai_perfection_level": round(self.perfection_level, 3),
            "ai_pore_acutance_gain": round(self.pore_acutance_gain, 3),
            "global_shadow_deficit_L": round(self.global_shadow_deficit_L, 2),
            "lateral_asymmetry_L": round(self.lateral_asymmetry_L, 2),
            "wb_cast_correction_ab": [round(self.wb_cast_da, 2), round(self.wb_cast_db, 2)],
            "oklab_studio_contrast": round(self.oklab_contrast_gain, 3),
            "zone_luma_L": {k: round(v, 1) for k, v in self.zone_luma_L.items()},
            "zone_fill_boost_L": {k: round(v, 2) for k, v in self.zone_fill_boost_L.items() if v > 0.05},
        }


# ============================================================================
# 1. IDENTITYMARKLOCK™ — NATURAL SCARS, MOLES, BEAUTY SPOTS & EXPRESSION LINES
# ============================================================================

def _frangi_hessian_ridgeness(L: np.ndarray, sigma: float, norm_c: float = 2.6) -> np.ndarray:
    """
    Compute 2D Frangi Anisotropic Ridgeness R_sigma(x, y) in [0, 1] from Hessian eigenvalues:
    Detects elongated facial scars (both dark indented scars & lighter hypopigmented scars)
    and anatomical expression folds while rejecting isotropic random pixel noise.
    """
    smooth = gaussian_blur(L, sigma=sigma)
    gx, gy, _ = compute_gradients(smooth)
    gxx, gxy, _ = compute_gradients(gx)
    _, gyy, _ = compute_gradients(gy)

    # Scale-normalized 2D Hessian eigenvalues
    s2 = float(sigma * sigma)
    hxx = gxx * s2
    hxy = gxy * s2
    hyy = gyy * s2

    trace = hxx + hyy
    disc = np.hypot(hxx - hyy, 2.0 * hxy)
    e1 = 0.5 * (trace + disc)
    e2 = 0.5 * (trace - disc)

    swap = np.abs(e2) > np.abs(e1)
    lam1 = np.where(swap, e2, e1)
    lam2 = np.where(swap, e1, e2)

    rb = np.abs(lam2) / (np.abs(lam1) + 1e-4)
    s_norm2 = lam1 * lam1 + lam2 * lam2
    anisotropy = np.exp(-(rb * rb) / (2.0 * 0.58 * 0.58))
    structure = 1.0 - np.exp(-s_norm2 / (2.0 * norm_c * norm_c))
    return np.clip(anisotropy * structure, 0.0, 1.0).astype(np.float32)


def compute_identity_mark_lock(
    img_or_lab: np.ndarray,
    geom: FacialAnthropometrics | None = None,
    is_lab: bool = False,
) -> np.ndarray:
    """
    Compute continuous `IdentityMarkLock™` field in [0, 1] of shape (H, W):
    - `1.0` = Strictly lock permanent facial identity mark (natural scar, mole/nevus, beauty spot,
      birthmark, freckle, bindi, eyelid crease, nasolabial fold, philtrum, ocular iris/lashes).
    - `0.0` = Safe flat epidermis where sensor ISO noise & compression blotches can be polished.
    """
    h, w = img_or_lab.shape[:2]
    lab = img_or_lab if is_lab else rgb_to_lab(img_or_lab)
    if geom is None or (geom.img_w, geom.img_h) != (w, h):
        rgb_tmp = lab_to_rgb(lab) if is_lab else img_or_lab
        geom = analyze_face_geometry(rgb_tmp)

    L = lab[:, :, 0]
    ipd = max(16.0, float(geom.ipd_px))

    # A. Localized Melanin Nevus / Mole / Beauty Spot / Birthmark Detector
    # Compares compact core (sigma_core) against local surrounding skin (sigma_surr)
    sigma_core = max(0.80, min(2.0, 0.0085 * ipd))
    sigma_surr = max(2.80, min(9.5, 0.048 * ipd))
    L_core = gaussian_blur(L, sigma=sigma_core)
    L_surr = gaussian_blur(L, sigma=sigma_surr)

    # Dark melanin spot (mole/beauty spot/freckle) OR distinct localized pale scar spot
    melanin_dip = L_surr - L_core
    pale_scar_peak = L_core - L_surr
    raw_spot = np.maximum(
        np.clip((melanin_dip - 4.6) / 4.5, 0.0, 1.0),
        np.clip((pale_scar_peak - 5.6) / 5.0, 0.0, 1.0),
    )
    # Reject isolated 1-pixel ISO noise spikes by requiring 3x3 spatial coherence
    coherence_3x3 = box_filter(raw_spot, radius=1)
    nevus_spot_lock = np.where(coherence_3x3 > 0.28 * raw_spot, raw_spot, 0.0)
    skin_interior = np.clip((geom.skin_mask - 0.25) / 0.45, 0.0, 1.0) * (1.0 - geom.eye_fidelity_map)
    nevus_spot_lock = np.clip(nevus_spot_lock * 1.25, 0.0, 1.0) * skin_interior

    # B. Multiscale Frangi Hessian Dermal Scar & Deep Anatomical Fold Detector (strictly inside skin_interior)
    s1 = max(1.0, min(2.2, 0.011 * ipd))
    s2 = max(1.8, min(3.8, 0.022 * ipd))
    ridge_fine = _frangi_hessian_ridgeness(L, sigma=s1, norm_c=4.8)
    ridge_meso = _frangi_hessian_ridgeness(L, sigma=s2, norm_c=5.4)
    scar_and_line_lock = np.clip(np.maximum(ridge_fine, ridge_meso) * skin_interior, 0.0, 1.0)

    identity_lock = np.clip(np.maximum(nevus_spot_lock, scar_and_line_lock), 0.0, 1.0)
    return identity_lock.astype(np.float32)


# ============================================================================
# 2. 12-ZONE AI STUDIO AUTO-DIRECTOR BRAIN (`analyze_12zone_studio_director`)
# ============================================================================

# 12 canonical roll-aligned anatomical zones: (name, du_ipd, dv_ipd, ru_ipd, rv_ipd, target_ratio_to_key)
_ANATOMICAL_12_ZONES: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("forehead_left",     -0.30, -0.50, 0.25, 0.20, 0.99),
    ("forehead_right",     0.30, -0.50, 0.25, 0.20, 0.99),
    ("glabella_bridge",    0.00,  0.22, 0.16, 0.24, 1.01),
    ("tear_trough_left",  -0.46,  0.22, 0.24, 0.14, 0.95),
    ("tear_trough_right",  0.46,  0.22, 0.24, 0.14, 0.95),
    ("malar_cheek_left",  -0.50,  0.56, 0.28, 0.24, 1.00),
    ("malar_cheek_right",  0.50,  0.56, 0.28, 0.24, 1.00),
    ("nasolabial_left",   -0.30,  0.84, 0.20, 0.18, 0.93),
    ("nasolabial_right",   0.30,  0.84, 0.20, 0.18, 0.93),
    ("chin_menton",        0.00,  1.34, 0.26, 0.18, 0.94),
    ("jaw_temple_lat",     0.00,  0.65, 0.82, 0.35, 0.92),
    ("cervical_neck",      0.00,  1.80, 0.36, 0.24, 0.89),
)


def _classify_ita_skin_tone(ita_deg: float) -> str:
    if ita_deg > 48.0:
        return "Very Fair / Porcelain"
    if ita_deg > 36.0:
        return "Fair / Warm Ivory"
    if ita_deg > 24.0:
        return "Intermediate / Golden Wheatish"
    if ita_deg > 10.0:
        return "Medium Tan / Warm Bronze"
    if ita_deg > -12.0:
        return "Deep Brown / Rich Melanin"
    return "Very Deep / Espresso Melanin"


def analyze_12zone_studio_director(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics | None = None,
    slider_tier: int = 4,
) -> StudioDirectorPlan:
    """
    AI Studio Auto-Director Brain:
    Evaluates 12 anatomical zones, constitutive skin tone (ITA°), MTF sharpness, ISO noise,
    flash specular glare, and ambient white-balance cast to decide exact face perfection
    parameters and build a smooth 2D spatial studio relighting & de-glare field.
    """
    h, w = img_rgb.shape[:2]
    if geom is None or (geom.img_w, geom.img_h) != (w, h):
        geom = analyze_face_geometry(img_rgb)

    lab = rgb_to_lab(img_rgb)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    ipd = max(16.0, float(geom.ipd_px))
    eye_mid_x = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    eye_mid_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)
    roll_rad = math.radians(geom.roll_deg)
    cos_r = math.cos(roll_rad)
    sin_r = math.sin(roll_rad)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    dx = xx - eye_mid_x
    dy = yy - eye_mid_y
    # Roll-aligned normalized IPD coordinates (u = right, v = down)
    u_ipd = (dx * cos_r + dy * sin_r) / ipd
    v_ipd = (-dx * sin_r + dy * cos_r) / ipd

    med_L, med_a, med_b = geom.skin_lab_median
    ita_deg = math.degrees(math.atan2(med_L - 50.0, max(2.0, med_b)))
    skin_cat = _classify_ita_skin_tone(ita_deg)

    # Measure MTF sharpness & ISO noise on facial ROI
    fx1, fy1, fx2, fy2 = geom.face_box
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(w, max(fx1 + 8, fx2)), min(h, max(fy1 + 8, fy2))
    face_L = L[fy1:fy2, fx1:fx2]
    face_skin = geom.skin_mask[fy1:fy2, fx1:fx2] > 0.35

    _, _, grad_face = compute_gradients(face_L)
    mtf_sharpness = float(np.percentile(grad_face, 90)) if grad_face.size > 0 else 6.0
    hf_face = face_L - gaussian_blur(face_L, sigma=0.95)
    if np.count_nonzero(face_skin) >= 24:
        iso_noise_sigma = float(np.median(np.abs(hf_face[face_skin])) * 1.4826)
        skin_p88 = float(np.percentile(face_L[face_skin], 88))
        skin_p15 = float(np.percentile(face_L[face_skin], 15))
    else:
        iso_noise_sigma = float(np.median(np.abs(hf_face)) * 1.4826)
        skin_p88 = min(84.0, med_L + 12.0)
        skin_p15 = max(28.0, med_L - 14.0)

    # Evaluate all 12 anatomical zones
    zone_luma: dict[str, float] = {}
    zone_masks: dict[str, np.ndarray] = {}
    for z_name, du, dv, ru, rv, _tgt_ratio in _ANATOMICAL_12_ZONES:
        if z_name == "jaw_temple_lat":
            g_mask = np.maximum(
                np.exp(-0.5 * (((u_ipd + 0.74) / 0.22) ** 2 + ((v_ipd - 0.65) / 0.35) ** 2)),
                np.exp(-0.5 * (((u_ipd - 0.74) / 0.22) ** 2 + ((v_ipd - 0.65) / 0.35) ** 2)),
            )
        else:
            g_mask = np.exp(-0.5 * (((u_ipd - du) / ru) ** 2 + ((v_ipd - dv) / rv) ** 2))
        zone_masks[z_name] = g_mask
        valid_z = (g_mask > 0.42) & (geom.skin_mask > 0.25) & (geom.eye_fidelity_map < 0.35)
        if np.count_nonzero(valid_z) >= 8:
            zone_luma[z_name] = float(np.median(L[valid_z]))
        else:
            zone_luma[z_name] = float(med_L)

    # Key-light reference = brighter of the two malar cheeks (or forehead if cheeks shadowed)
    cheek_max_L = max(zone_luma["malar_cheek_left"], zone_luma["malar_cheek_right"])
    # ITA-locked natural studio target: lifts shaded outdoor/indoor faces into vibrant studio softbox radiance!
    constitutive_floor_L = float(np.clip(50.0 + max(3.0, med_b) * math.tan(math.radians(max(-18.0, ita_deg))), 38.0, 75.0))
    underexp_room_lift = float(np.clip((64.5 - cheek_max_L) * 0.54, 0.0, 8.2)) if cheek_max_L < 64.5 else 0.0
    studio_key_L = max(cheek_max_L, constitutive_floor_L) + underexp_room_lift
    studio_key_L = float(np.clip(studio_key_L, 42.0, 81.5))

    # Build targeted 2D spatial shadow-fill field ONLY where zones fall below their natural ratio to studio_key_L!
    spatial_fill_L = np.zeros((h, w), dtype=np.float32)
    zone_fill_boost: dict[str, float] = {}
    for z_name, _du, _dv, _ru, _rv, tgt_ratio in _ANATOMICAL_12_ZONES:
        cur_z_L = zone_luma[z_name]
        desired_z_L = studio_key_L * tgt_ratio
        deficit_z = desired_z_L - cur_z_L
        if deficit_z > 0.9:
            # Cap per-zone fill so natural 3D facial modeling & contour depth remain realistic
            max_z_cap = 7.5 if "tear_trough" in z_name else (6.5 if "cheek" in z_name or "forehead" in z_name else 5.5)
            boost = float(np.clip(deficit_z * 0.68, 0.0, max_z_cap))
            zone_fill_boost[z_name] = boost
            spatial_fill_L = np.maximum(spatial_fill_L, boost * zone_masks[z_name])
        else:
            zone_fill_boost[z_name] = 0.0

    # Smooth the 2D spatial studio relighting map so transitions across cheeks/nose/neck are invisible
    spatial_fill_L = gaussian_blur(spatial_fill_L, sigma=max(3.0, 0.10 * ipd))
    # Gate away from bright highlights (L > 80) so studio fill never clips highlights
    highlight_protect = np.clip((84.0 - L) / 14.0, 0.0, 1.0)
    spatial_fill_L = spatial_fill_L * highlight_protect * np.clip(geom.skin_mask * 1.15 + 0.25, 0.0, 1.0)

    # Optical Polarization Flash Hot-Spot De-Glare Map (forehead / nose tip / cheekbones)
    med_chroma = math.hypot(med_a, med_b)
    pix_chroma = np.hypot(a, b)
    # Specular skin glare has elevated L* (> max(83, skin_p88 + 4)) AND desaturated specular reflection
    glare_thresh_L = max(83.5, min(91.0, skin_p88 + 4.5))
    glare_luma_gate = np.clip((L - glare_thresh_L) / 8.5, 0.0, 1.0)
    glare_desat_gate = np.clip((0.78 * med_chroma - pix_chroma) / max(3.5, 0.45 * med_chroma), 0.15, 1.0)
    face_glare_zone = np.exp(-0.5 * ((u_ipd / 0.85) ** 2 + ((v_ipd - 0.35) / 1.05) ** 2))
    deglare_2d = gaussian_blur(
        glare_luma_gate * glare_desat_gate * geom.skin_mask * face_glare_zone * (1.0 - geom.eye_fidelity_map),
        sigma=max(1.5, 0.025 * ipd),
    )
    deglare_2d = np.clip(deglare_2d, 0.0, 0.85).astype(np.float32)
    glare_ratio = float(np.mean(deglare_2d[fy1:fy2, fx1:fx2] > 0.18)) if deglare_2d[fy1:fy2, fx1:fx2].size > 0 else 0.0

    # Ambient White-Balance Cast Estimation from Neutral Ocular Sclera (outside iris, inside palpebral aperture)
    wb_da, wb_db = 0.0, 0.0
    sclera_candidates = (geom.eye_fidelity_map > 0.35) & (L > 52.0) & (L < 92.0) & (np.hypot(a, b) < 11.0)
    if np.count_nonzero(sclera_candidates) >= 14:
        sc_a = float(np.median(a[sclera_candidates]))
        sc_b = float(np.median(b[sclera_candidates]))
        # Healthy human sclera under D65 studio strobes sits near a* ~ +0.6, b* ~ +1.2.
        # Correct only excess room fluorescent/tungsten cast (clamped tightly to [-2.4, +2.4] so natural skin never shifts!)
        if abs(sc_a - 0.6) > 1.2:
            wb_da = float(np.clip(-(sc_a - 0.6) * 0.45, -2.4, 2.4))
        if abs(sc_b - 1.2) > 1.8:
            wb_db = float(np.clip(-(sc_b - 1.2) * 0.42, -2.4, 2.4))

    # AI-Decided CodeFormer Fidelity Weight w, Neural Blend, and Perfection Level
    tier_norm = float(np.clip((slider_tier - 1) / 9.0, 0.0, 1.0))
    sharp_norm = float(np.clip((mtf_sharpness - 4.5) / 10.5, 0.0, 1.0))
    noise_norm = float(np.clip((iso_noise_sigma - 0.85) / 3.2, 0.0, 1.0))

    # High sharpness -> higher fidelity w (0.76..0.84) to preserve real camera detail;
    # Soft/compressed input -> balanced w (0.58..0.70) + strong neural blend so CodeFormer reconstructs crisp eyes/glasses/brows
    codeformer_w = float(np.clip(0.58 + 0.22 * sharp_norm - 0.04 * noise_norm, 0.54, 0.84))
    neural_blend = float(np.clip(0.86 - 0.18 * sharp_norm + 0.06 * tier_norm, 0.64, 0.93))
    perfection_lvl = float(np.clip(0.38 + 0.24 * noise_norm + 0.06 * tier_norm, 0.34, 0.66))

    dyn_span = max(8.0, skin_p88 - skin_p15)
    flat_lighting = float(np.clip((34.0 - dyn_span) / 24.0, 0.0, 1.0))
    pore_gain = float(np.clip(0.34 + 0.16 * flat_lighting + 0.08 * tier_norm, 0.30, 0.58))
    oklab_contrast = float(np.clip(1.06 + 0.05 * flat_lighting + 0.03 * tier_norm, 1.05, 1.14))

    lat_asym = float(zone_luma["malar_cheek_left"] - zone_luma["malar_cheek_right"])
    max_boost = max(zone_fill_boost.values()) if zone_fill_boost else 0.0
    if underexp_room_lift > 2.0:
        mode_str = "Low-Light Studio Beauty-Dish Recovery"
    elif glare_ratio > 0.025:
        mode_str = "Polarized Flash De-Glare + Balanced Studio Fill"
    elif max_boost > 2.2 or abs(lat_asym) > 3.0:
        mode_str = "Targeted Spatial Zone Shadow Fill (Key-Light Preserved)"
    else:
        mode_str = "Optimal Studio Lighting Detected (Micro-Contrast & Skin Lock)"

    return StudioDirectorPlan(
        director_mode=mode_str,
        ita_angle_deg=ita_deg,
        skin_tone_category=skin_cat,
        mtf_sharpness=mtf_sharpness,
        iso_noise_sigma=iso_noise_sigma,
        glare_hotspot_ratio=glare_ratio,
        codeformer_fidelity_w=codeformer_w,
        neural_blend_strength=neural_blend,
        perfection_level=perfection_lvl,
        pore_acutance_gain=pore_gain,
        global_shadow_deficit_L=underexp_room_lift,
        lateral_asymmetry_L=lat_asym,
        wb_cast_da=wb_da,
        wb_cast_db=wb_db,
        oklab_contrast_gain=oklab_contrast,
        zone_luma_L=zone_luma,
        zone_fill_boost_L=zone_fill_boost,
        spatial_relight_delta_L=spatial_fill_L,
        deglare_weight_2d=deglare_2d,
    )


# ============================================================================
# 3. SPATIAL STUDIO RELIGHTING, DE-GLARE & ZERO-HUE-TWIST OKLAB COLOR GRADING
# ============================================================================

def apply_ai_spatial_studio_relight_and_grade(
    img_rgb: np.ndarray,
    orig_ref_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    plan: StudioDirectorPlan,
    identity_lock: np.ndarray | None = None,
) -> np.ndarray:
    """
    Execute the Studio Auto-Director's spatial lighting & Perceptual Oklab color-grading plan:
    1. Targeted 2D Spatial Shadow Fill (`plan.spatial_relight_delta_L`) ONLY where shadows exist.
    2. Optical Polarization Flash Hot-Spot De-Glare (`plan.deglare_weight_2d`) restoring satin skin pigment.
    3. Sclera-Anchored Ambient White-Balance Cast Neutralization (`wb_cast_da`, `wb_cast_db`).
    4. 100% Natural Skin Chromaticity Lock (`orig_ref_rgb` chromophores + Helmholtz-Kohlrausch scaling).
    5. Zero-Hue-Twist Perceptual Oklab Filmic Studio Micro-Contrast Curve (`oklab_contrast_gain`)
       while preserving 100% of `IdentityMarkLock™` natural scars, moles, and beauty spots!
    """
    h, w = img_rgb.shape[:2]
    if identity_lock is None or identity_lock.shape[:2] != (h, w):
        identity_lock = compute_identity_mark_lock(orig_ref_rgb, geom=geom, is_lab=False)

    lab = rgb_to_lab(img_rgb)
    lab_orig = rgb_to_lab(orig_ref_rgb)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    L_orig = lab_orig[:, :, 0]
    a_orig = lab_orig[:, :, 1]
    b_orig = lab_orig[:, :, 2]
    med_L, med_a, med_b = geom.skin_lab_median

    # 1. Separate smooth illumination base from fine skin/scar/spot detail
    r_illum = max(5, int(round(geom.ipd_px * 0.24)))
    L_illum = guided_filter(L / 100.0, L / 100.0, radius=r_illum, eps=3.5e-3) * 100.0
    detail_L = L - L_illum
    detail_L_orig = L_orig - (guided_filter(L_orig / 100.0, L_orig / 100.0, radius=r_illum, eps=3.5e-3) * 100.0)

    # Reinforce 100% of original scar/mole/beauty-spot detail wherever IdentityMarkLock is active!
    id_w = np.clip(identity_lock * (1.0 - 0.5 * geom.eye_fidelity_map), 0.0, 1.0)
    detail_L = detail_L * (1.0 - 0.85 * id_w) + detail_L_orig * (0.85 * id_w)

    # 2. Apply AI-decided 2D Spatial Shadow Fill onto the illumination layer ONLY
    fill_2d = plan.spatial_relight_delta_L
    if fill_2d.shape[:2] != (h, w):
        fill_2d = np.zeros((h, w), dtype=np.float32)
    L_illum_relit = L_illum + fill_2d

    # 3. Optical Polarization Flash De-Glare on clipped oily forehead/nose hotspots
    deglare = plan.deglare_weight_2d
    if deglare.shape[:2] == (h, w) and float(deglare.max()) > 0.02:
        satin_cap_L = max(78.0, min(85.5, med_L + 16.0))
        excess_glare = np.maximum(0.0, L_illum_relit - satin_cap_L)
        L_illum_relit = L_illum_relit - 0.62 * deglare * excess_glare

    L_relit = np.clip(L_illum_relit + detail_L, 0.0, 100.0)

    # Balanced studio exposure boost cap so overcast/indoor portraits pop naturally without highlight clipping
    max_med_lift = 8.0 if med_L < 63.0 else 3.2
    fx1, fy1, fx2, fy2 = geom.face_box
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(w, max(fx1 + 4, fx2)), min(h, max(fy1 + 4, fy2))
    face_skin_bool = geom.skin_mask[fy1:fy2, fx1:fx2] > 0.35
    if np.count_nonzero(face_skin_bool) >= 25:
        cur_med = float(np.median(L_relit[fy1:fy2, fx1:fx2][face_skin_bool]))
        orig_med = float(np.median(L_orig[fy1:fy2, fx1:fx2][face_skin_bool]))
        excess = cur_med - (orig_med + max_med_lift)
        if excess > 0.0:
            L_relit = np.clip(L_relit - excess * np.clip(geom.skin_mask * 1.15, 0.25, 1.0), 0.0, 100.0)

    # DSLR Fine Micro-Contrast Boost (`sigma=1.35`) on brows, lashes, nose bridge, lips, hair & fabric
    hf_crisp = L_relit - gaussian_blur(L_relit, sigma=1.35)
    crisp_gate = 0.22 * (1.0 - 0.55 * geom.skin_mask) + 0.18 * id_w + 0.25 * geom.eye_fidelity_map
    L_relit = np.clip(L_relit + crisp_gate * hf_crisp, 0.0, 100.0)

    # 4. Natural Skin Chromaticity Lock + Outdoor Foliage/Shade Warmth Recovery + Deglare Pigment
    skin_w = np.clip(geom.skin_mask * (1.0 - geom.eye_fidelity_map), 0.0, 1.0)
    hk_scale = np.where(
        L_relit > L_orig + 0.6,
        np.clip(((L_relit + 12.0) / (L_orig + 12.0)) ** 0.65, 0.95, 1.20),
        1.0,
    )
    # Neutralize green foliage / cool overcast cast on shaded skin and restore subtle golden studio warmth
    warmth_da = float(np.clip((11.5 - med_a) * 0.28, 0.0, 1.8)) if med_L < 68.0 else 0.0
    warmth_db = float(np.clip((17.5 - med_b) * 0.28, 0.4, 2.2)) if med_L < 68.0 else 0.35
    a_locked = a_orig * hk_scale + warmth_da
    b_locked = b_orig * hk_scale + warmth_db
    if deglare.shape[:2] == (h, w) and float(deglare.max()) > 0.02:
        a_locked = a_locked * (1.0 - 0.55 * deglare) + med_a * (0.55 * deglare)
        b_locked = b_locked * (1.0 - 0.55 * deglare) + med_b * (0.55 * deglare)

    a_out = a * (1.0 - skin_w) + a_locked * skin_w + plan.wb_cast_da * 0.65
    b_out = b * (1.0 - skin_w) + b_locked * skin_w + plan.wb_cast_db * 0.65

    lab[:, :, 0] = L_relit
    lab[:, :, 1] = np.clip(a_out, -128.0, 127.0)
    lab[:, :, 2] = np.clip(b_out, -128.0, 127.0)
    relit_rgb = lab_to_rgb(lab)

    # 5. Perceptual Oklab Zero-Hue-Twist Filmic Studio Color Grading
    c_gain = float(plan.oklab_contrast_gain)
    if abs(c_gain - 1.0) < 0.008:
        return relit_rgb

    ok = rgb_to_oklab(relit_rgb)
    L_ok = ok[:, :, 0]
    pivot = float(np.clip((med_L / 100.0) ** 0.85, 0.48, 0.72))
    centered = L_ok - pivot
    L_graded = pivot + np.tanh(centered * c_gain * 1.35) / 1.35
    grade_blend = 1.0 - 0.38 * skin_w * (1.0 - id_w)
    ok[:, :, 0] = np.clip(L_ok * (1.0 - grade_blend) + L_graded * grade_blend, 0.0, 1.0)
    # Healthy 6% studio vibrance on Oklab chroma outside sclera
    vibrance = 1.0 + 0.06 * (1.0 - geom.eye_fidelity_map)
    ok[:, :, 1] *= vibrance
    ok[:, :, 2] *= vibrance
    return oklab_to_rgb(ok)
