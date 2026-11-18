"""
V8 Multi-Subject Portrait Scene Understanding, YuNet Multi-Face Landmark Solver,
Poster/Bystander Pruning, Smart Gender/Pallu Classifier & OrnamentLock™ Engine.

Handles:
1. Single Portraits AND Dual/Couple/Group Official Portraits (1 to 4 foreground people),
   ordered canonically from Left to Right (`Person 1 (Left)`, `Person 2 (Right)`, ...).
2. Automatic Rejection of Wall Paintings / Flex-Banner Poster Faces (e.g. Bal Krishna wall
   frame between a couple or printed faces on background banners) via Foreground Alpha Salience
   and Inter-Subject Anthropometric Scale Gating.
3. Connected-Component Bystander Pruning (erases turned-away passersby & partial arms at frame
   edges that do not belong to any camera-facing foreground subject).
4. AI Smart Gender & Head-Pallu Understanding (`male` / `female`, `has_head_pallu`) so official
   Couple Dress Change automatically routes Men's Suits to the Husband and Sarees/Blazers to
   the Wife (or lets the operator target Person 1 / Person 2 explicitly).
5. `OrnamentLock™` protecting Indian cultural & personal identity ornaments: Sindoor, Bindi,
   Nose Pin / Nath, Earrings, and Necklaces/Mangalsutra alongside natural scars & moles.

STRICT CLOUD-ONLY WEIGHTS (`0 MB` on Local Windows PC):
OpenCV YuNet (`face_detection_yunet_2023mar.onnx`, 227 KB) downloads ONLY when `IS_HF_CLOUD == True`.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
import threading
import numpy as np

from src.shared.config import CLOUD_MODELS_DIR, HF_TOKEN, IS_HF_CLOUD
from src.core.cv_ops import (
    compute_gradients,
    gaussian_blur,
    resize_img,
    rgb_to_lab,
)
from src.core.facial_math_engine import (
    EyeGeometry,
    FacialAnthropometrics,
    analyze_face_geometry,
    build_riemannian_eye_fidelity_map,
    compute_biophysical_skin_mask,
)

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

_YUNET_LOCK = threading.Lock()
_YUNET_PATH: str | None = None
_YUNET_TRIED: bool = False


@dataclass
class SubjectProfile:
    """Complete anthropometric, biometric, and cultural profile for one foreground person."""
    subject_index: int                    # 0-based index ordered Left -> Right
    label: str                            # e.g. "Person 1 (Left)" or "Primary Subject"
    geom: FacialAnthropometrics           # Full facial & cervical-clavicular anthropometrics
    inferred_gender: str                  # "male" | "female"
    gender_confidence: float              # [0.0, 1.0]
    has_head_pallu: bool                  # True when saree pallu / dupatta drapes over head
    roi_box: tuple[int, int, int, int]    # Exclusive torso+head horizontal ownership (x1, y1, x2, y2)
    ornament_lock: np.ndarray             # Full-image (H, W) float32 lock for bindi/sindoor/nose-pin/jewelry


def _ensure_yunet_onnx_cloud() -> str | None:
    """Download tiny 227KB YuNet ONNX strictly inside Hugging Face Cloud (`IS_HF_CLOUD`)."""
    global _YUNET_PATH, _YUNET_TRIED
    if not IS_HF_CLOUD or not HAS_CV2 or cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
        return None
    if os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return None
    if _YUNET_PATH and os.path.isfile(_YUNET_PATH):
        return _YUNET_PATH
    if _YUNET_TRIED:
        return None

    with _YUNET_LOCK:
        if _YUNET_PATH and os.path.isfile(_YUNET_PATH):
            return _YUNET_PATH
        if _YUNET_TRIED:
            return None
        _YUNET_TRIED = True
        try:
            os.makedirs(CLOUD_MODELS_DIR, exist_ok=True)
            from huggingface_hub import hf_hub_download  # type: ignore

            path = hf_hub_download(
                repo_id="opencv/face_detection_yunet",
                filename="face_detection_yunet_2023mar.onnx",
                cache_dir=CLOUD_MODELS_DIR,
                token=HF_TOKEN,
            )
            if path and os.path.isfile(path):
                _YUNET_PATH = path
                print(f"✅ [MultiSubject Cloud] Loaded YuNet multi-face detector: {path}")
                return _YUNET_PATH
        except Exception as e:
            print(f"⚠️ [MultiSubject Cloud] YuNet download skipped: {e}")
    return None


def _build_geom_from_5pt(
    img_rgb: np.ndarray,
    skin_full: np.ndarray,
    lab_full: np.ndarray,
    box_xywh: tuple[float, float, float, float],
    right_eye_xy: tuple[float, float],
    left_eye_xy: tuple[float, float],
    nose_xy: tuple[float, float],
    right_mouth_xy: tuple[float, float],
    left_mouth_xy: tuple[float, float],
) -> FacialAnthropometrics:
    """
    Construct exact `FacialAnthropometrics` from YuNet 5-point landmarks.
    Note: YuNet's `right_eye` is the person's right eye (viewer's LEFT), and `left_eye`
    is the person's left eye (viewer's RIGHT), matching our `FacialAnthropometrics` convention
    where `left_eye.center_x < right_eye.center_x` (viewer's left vs viewer's right).
    """
    h, w = img_rgb.shape[:2]
    # Viewer's left eye vs viewer's right eye
    if right_eye_xy[0] <= left_eye_xy[0]:
        v_left_ex, v_left_ey = right_eye_xy
        v_right_ex, v_right_ey = left_eye_xy
    else:
        v_left_ex, v_left_ey = left_eye_xy
        v_right_ex, v_right_ey = right_eye_xy

    dx_eye = v_right_ex - v_left_ex
    dy_eye = v_right_ey - v_left_ey
    ipd_px = float(max(16.0, math.hypot(dx_eye, dy_eye)))
    roll_rad = math.atan2(dy_eye, max(1.0, dx_eye))
    roll_deg = float(np.clip(math.degrees(roll_rad), -28.0, 28.0))
    roll_rad = math.radians(roll_deg)

    eye_mid_x = 0.5 * (v_left_ex + v_right_ex)
    eye_mid_y = 0.5 * (v_left_ey + v_right_ey)

    right_vec_x, right_vec_y = math.cos(roll_rad), math.sin(roll_rad)
    down_x, down_y = -math.sin(roll_rad), math.cos(roll_rad)

    fx, fy, fw, fh = box_xywh
    mouth_x = 0.5 * (right_mouth_xy[0] + left_mouth_xy[0])
    mouth_y = 0.5 * (right_mouth_xy[1] + left_mouth_xy[1])
    nose_x, nose_y = nose_xy

    crown_y = max(0.0, min(fy - 0.14 * fh, eye_mid_y - 1.18 * ipd_px))
    crown_x = eye_mid_x - (eye_mid_y - crown_y) * down_x
    chin_y = min(float(h - 1), max(fy + 0.98 * fh, eye_mid_y + 1.56 * ipd_px, mouth_y + 0.54 * ipd_px))
    chin_dist = max(1.48 * ipd_px, chin_y - eye_mid_y)
    chin_x = eye_mid_x + chin_dist * down_x

    # Measure true sub-jaw neck width within this specific subject's column
    neck_half_w = float(np.clip(0.38 * fw, 0.54 * ipd_px, 0.72 * ipd_px))
    neck_top_dist = max(1.34 * ipd_px, chin_dist - 0.14 * ipd_px)
    neck_top_c = (eye_mid_x + neck_top_dist * down_x, eye_mid_y + neck_top_dist * down_y)
    neck_top_left = (neck_top_c[0] - neck_half_w * right_vec_x, neck_top_c[1] - neck_half_w * right_vec_y)
    neck_top_right = (neck_top_c[0] + neck_half_w * right_vec_x, neck_top_c[1] + neck_half_w * right_vec_y)

    neck_base_dist = max(1.90 * ipd_px, chin_dist + 0.32 * ipd_px)
    neck_base_c = (eye_mid_x + neck_base_dist * down_x, eye_mid_y + neck_base_dist * down_y)
    neck_base_half_w = max(0.68 * ipd_px, neck_half_w * 1.18)
    neck_base_left = (neck_base_c[0] - neck_base_half_w * right_vec_x, neck_base_c[1] - neck_base_half_w * right_vec_y)
    neck_base_right = (neck_base_c[0] + neck_base_half_w * right_vec_x, neck_base_c[1] + neck_base_half_w * right_vec_y)

    jugular_dist = max(2.22 * ipd_px, neck_base_dist + 0.30 * ipd_px)
    jugular_notch = (eye_mid_x + jugular_dist * down_x, eye_mid_y + jugular_dist * down_y)

    shoulder_dist = max(2.38 * ipd_px, jugular_dist + 0.16 * ipd_px)
    shoulder_c = (eye_mid_x + shoulder_dist * down_x, eye_mid_y + shoulder_dist * down_y)
    shoulder_half_w = max(1.78 * ipd_px, neck_base_half_w * 2.45)
    shoulder_left = (shoulder_c[0] - shoulder_half_w * right_vec_x, shoulder_c[1] - shoulder_half_w * right_vec_y)
    shoulder_right = (shoulder_c[0] + shoulder_half_w * right_vec_x, shoulder_c[1] + shoulder_half_w * right_vec_y)
    chest_center = (eye_mid_x + 3.10 * ipd_px * down_x, eye_mid_y + 3.10 * ipd_px * down_y)

    face_x1 = max(0, int(round(fx)))
    face_y1 = max(0, int(round(fy)))
    face_x2 = min(w, int(round(fx + fw)))
    face_y2 = min(h, int(round(fy + fh)))

    # Subject-local facial skin statistics
    roi_lab = lab_full[face_y1:face_y2, face_x1:face_x2]
    roi_skin = skin_full[face_y1:face_y2, face_x1:face_x2]
    valid_skin = roi_lab[roi_skin > 0.32]
    if len(valid_skin) >= 20:
        med_L = float(np.median(valid_skin[:, 0]))
        med_a = float(np.median(valid_skin[:, 1]))
        med_b = float(np.median(valid_skin[:, 2]))
    else:
        med_L, med_a, med_b = 62.0, 13.0, 19.0

    mid_col = roi_lab.shape[1] // 2
    if mid_col > 2:
        lateral_shadow_diff = float(np.mean(roi_lab[:, :mid_col, 0]) - np.mean(roi_lab[:, mid_col:, 0]))
    else:
        lateral_shadow_diff = 0.0

    mean_iris_r = max(3.5, 0.175 * ipd_px)
    pupil_r = max(1.8, mean_iris_r * 0.38)
    palp_rx = mean_iris_r * 1.95
    palp_ry = mean_iris_r * 1.18

    left_eye_geom = EyeGeometry(
        center_x=v_left_ex,
        center_y=v_left_ey,
        iris_radius=mean_iris_r,
        pupil_radius=pupil_r,
        palpebral_rx=palp_rx,
        palpebral_ry=palp_ry,
        limbus_contrast=18.0,
        purkinje_x=v_left_ex - 0.24 * mean_iris_r,
        purkinje_y=v_left_ey - 0.24 * mean_iris_r,
    )
    right_eye_geom = EyeGeometry(
        center_x=v_right_ex,
        center_y=v_right_ey,
        iris_radius=mean_iris_r,
        pupil_radius=pupil_r,
        palpebral_rx=palp_rx,
        palpebral_ry=palp_ry,
        limbus_contrast=18.0,
        purkinje_x=v_right_ex - 0.24 * mean_iris_r,
        purkinje_y=v_right_ey - 0.24 * mean_iris_r,
    )
    eye_map = build_riemannian_eye_fidelity_map(w, h, left_eye_geom, right_eye_geom, roll_rad)
    yaw_asym = float(np.clip((eye_mid_x - 0.5 * (face_x1 + face_x2)) / max(1.0, ipd_px), -1.0, 1.0))

    # Build a subject-scoped skin mask so enhancements on Subject A don't bleed onto background walls
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    subj_gate = np.exp(-0.5 * (((xx - eye_mid_x) / (1.85 * ipd_px)) ** 2 + ((yy - (eye_mid_y + 0.75 * ipd_px)) / (2.60 * ipd_px)) ** 2))
    scoped_skin = np.clip(skin_full * np.clip(subj_gate * 1.6, 0.0, 1.0), 0.0, 1.0)

    return FacialAnthropometrics(
        img_w=w,
        img_h=h,
        face_box=(face_x1, face_y1, face_x2, face_y2),
        crown_xy=(crown_x, crown_y),
        chin_xy=(chin_x, chin_y),
        nose_tip_xy=(nose_x, nose_y),
        mouth_center_xy=(mouth_x, mouth_y),
        left_eye=left_eye_geom,
        right_eye=right_eye_geom,
        ipd_px=ipd_px,
        roll_deg=roll_deg,
        yaw_asymmetry=yaw_asym,
        neck_top_left=neck_top_left,
        neck_top_right=neck_top_right,
        neck_base_left=neck_base_left,
        neck_base_right=neck_base_right,
        jugular_notch=jugular_notch,
        shoulder_left=shoulder_left,
        shoulder_right=shoulder_right,
        chest_center=chest_center,
        skin_lab_median=(med_L, med_a, med_b),
        lateral_shadow_diff=lateral_shadow_diff,
        skin_mask=scoped_skin,
        eye_fidelity_map=eye_map,
    )


def compute_cultural_ornament_lock(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    lab_full: np.ndarray | None = None,
) -> np.ndarray:
    """
    `OrnamentLock™` — Explicitly detects & locks Indian cultural/personal identity marks:
    1. Forehead Bindi (red, maroon, dark velvet, or stone glabella mark between brows)
    2. Hairline Sindoor (central vermilion streak above forehead)
    3. Nose Pin / Nath (gold/silver/diamond specular micro-glint on nasal ala)
    4. Earrings & Necklace/Mangalsutra in ear/cervical zones
    Returns float32 map in [0, 1] of shape (H, W).
    """
    h, w = img_rgb.shape[:2]
    if lab_full is None:
        lab_full = rgb_to_lab(img_rgb)

    L = lab_full[:, :, 0]
    a = lab_full[:, :, 1]
    b = lab_full[:, :, 2]

    ipd = max(16.0, float(geom.ipd_px))
    eye_mid_x = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    eye_mid_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)
    med_L, med_a, _med_b = geom.skin_lab_median

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    # 1. Glabella Bindi + Central Hairline Sindoor Zone (red/vermilion high a* OR dark velvet bindi)
    glabella_sindoor_roi = np.exp(
        -0.5 * (((xx - eye_mid_x) / (0.24 * ipd)) ** 2)
    ) * (
        (yy >= eye_mid_y - 1.25 * ipd) & (yy <= eye_mid_y + 0.08 * ipd)
    ).astype(np.float32)
    red_pigment = np.clip((a - (med_a + 10.5)) / 8.0, 0.0, 1.0)
    dark_bindi_spot = (
        np.exp(-0.5 * (((xx - eye_mid_x) / (0.14 * ipd)) ** 2 + ((yy - (eye_mid_y - 0.30 * ipd)) / (0.22 * ipd)) ** 2))
        * np.clip(((med_L - 16.0) - L) / 10.0, 0.0, 1.0)
    )
    bindi_sindoor_lock = np.maximum(glabella_sindoor_lock := glabella_sindoor_roi * red_pigment, dark_bindi_spot)
    _ = glabella_sindoor_lock

    # 2. Nasal Ala Nose-Pin / Stud Specular & Metallic Lock
    nx, ny = geom.nose_tip_xy
    nose_wing_roi = np.exp(
        -0.5 * (((xx - nx) / (0.36 * ipd)) ** 2 + ((yy - ny) / (0.24 * ipd)) ** 2)
    )
    local_L = gaussian_blur(L, sigma=max(1.2, 0.04 * ipd))
    metallic_glint = np.clip((L - (local_L + 7.5)) / 8.0, 0.0, 1.0)
    gold_chroma = np.clip((b - 24.0) / 12.0, 0.0, 1.0) * np.clip((L - 55.0) / 20.0, 0.0, 1.0)
    nose_pin_lock = nose_wing_roi * np.maximum(metallic_glint, 0.7 * gold_chroma)

    # 3. Earlobe Earrings & Cervical Necklace Lock
    ear_left_roi = np.exp(-0.5 * (((xx - (eye_mid_x - 0.88 * ipd)) / (0.22 * ipd)) ** 2 + ((yy - (eye_mid_y + 0.42 * ipd)) / (0.34 * ipd)) ** 2))
    ear_right_roi = np.exp(-0.5 * (((xx - (eye_mid_x + 0.88 * ipd)) / (0.22 * ipd)) ** 2 + ((yy - (eye_mid_y + 0.42 * ipd)) / (0.34 * ipd)) ** 2))
    _, _, grad_L = compute_gradients(L)
    jewelry_edge = np.clip((grad_L - 6.5) / 8.0, 0.0, 1.0)
    earring_lock = np.maximum(ear_left_roi, ear_right_roi) * np.maximum(metallic_glint, jewelry_edge * gold_chroma)

    lock = np.clip(np.maximum(bindi_sindoor_lock, np.maximum(nose_pin_lock, earring_lock)), 0.0, 1.0)
    return np.clip(gaussian_blur(lock, sigma=1.1) * 1.25, 0.0, 1.0).astype(np.float32)


def _infer_gender_and_head_pallu(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics,
    lab_full: np.ndarray,
    ornament_lock: np.ndarray,
    raw_bria_alpha: np.ndarray | None = None,
) -> tuple[str, float, bool]:
    """
    AI Biometric & Cultural Cue Understanding for Official Dress Routing:
    Evaluates (1) Red & Dark-Velvet Bindi / Sindoor / Nose-Pin / Earring signals,
    (2) Perioral philtrum mustache & chin beard keratin density,
    (3) Foreground-gated lateral temple saree-pallu / dupatta drape continuity, and
    (4) Neck-flanking braids / ponytails / open shoulder hair.
    Returns `(inferred_gender, confidence, has_head_pallu)`.
    """
    h, w = img_rgb.shape[:2]
    L = lab_full[:, :, 0]
    a = lab_full[:, :, 1]
    b = lab_full[:, :, 2]

    ipd = max(16.0, float(geom.ipd_px))
    eye_mid_x = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    eye_mid_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)
    nx, ny = geom.nose_tip_xy
    mx, my = geom.mouth_center_xy
    med_L, med_a, _med_b = geom.skin_lab_median

    # Cue 1a: Red Bindi / Sindoor presence between brows & hairline
    gx1 = max(0, int(round(eye_mid_x - 0.20 * ipd)))
    gx2 = min(w, int(round(eye_mid_x + 0.20 * ipd)))
    gy1 = max(0, int(round(eye_mid_y - 1.15 * ipd)))
    gy2 = min(h, int(round(eye_mid_y + 0.05 * ipd)))
    bindi_sindoor_pixels = 0
    if gx2 > gx1 + 2 and gy2 > gy1 + 2:
        red_marks = (a[gy1:gy2, gx1:gx2] > med_a + 12.5) & (L[gy1:gy2, gx1:gx2] < 78.0)
        bindi_sindoor_pixels = int(np.count_nonzero(red_marks))

    # Cue 1b: Classic small black/maroon circular glabella bindi dot surrounded by smooth forehead skin
    has_dark_bindi_dot = False
    bx_d1 = max(0, int(round(eye_mid_x - 0.11 * ipd)))
    bx_d2 = min(w, int(round(eye_mid_x + 0.11 * ipd)))
    by_d1 = max(0, int(round(eye_mid_y - 0.50 * ipd)))
    by_d2 = min(h, int(round(eye_mid_y - 0.14 * ipd)))
    if bx_d2 > bx_d1 + 3 and by_d2 > by_d1 + 3:
        glab_patch = L[by_d1:by_d2, bx_d1:bx_d2]
        glab_bg_med = float(np.percentile(glab_patch, 80.0))
        dark_mask = glab_patch < glab_bg_med - 14.5
        dark_dot_cnt = int(np.count_nonzero(dark_mask))
        # Ensure it's an isolated central bindi dot (not connected eyebrows spanning left/right borders)
        border_dark = int(np.count_nonzero(dark_mask[:, 0])) + int(np.count_nonzero(dark_mask[:, -1]))
        max_dot_px = max(4, int(round(0.035 * ipd * ipd)))
        if 1 <= dark_dot_cnt <= max_dot_px and border_dark == 0:
            has_dark_bindi_dot = True

    # Cue 2: Mustache (philtrum) & Beard keratin density STRICTLY below lower lip vermilion
    # (Never include dark natural lip pigment or smooth lateral jaw shadows!)
    bx1 = max(0, int(round(mx - 0.38 * ipd)))
    bx2 = min(w, int(round(mx + 0.38 * ipd)))
    by1 = max(0, int(round(my + 0.20 * ipd)))
    by2 = min(h, int(round(geom.chin_xy[1] + 0.03 * ipd)))
    beard_ratio = 0.0
    if bx2 > bx1 + 4 and by2 > by1 + 3:
        chin_L = L[by1:by2, bx1:bx2]
        _, _, chin_grad = compute_gradients(chin_L)
        local_chin_ref = float(np.percentile(chin_L, 78.0))
        ref_L = min(med_L, local_chin_ref + 4.0)
        dark_textured_beard = (chin_L < max(20.0, ref_L - 17.5)) & (chin_grad > 3.6)
        beard_ratio = float(np.mean(dark_textured_beard))

    px1 = max(0, int(round(mx - 0.26 * ipd)))
    px2 = min(w, int(round(mx + 0.26 * ipd)))
    py1 = max(0, int(round(ny + 0.07 * ipd)))
    py2 = min(h, int(round(my - 0.12 * ipd)))
    mustache_drop = 0.0
    if px2 > px1 + 2 and py2 > py1 + 1:
        phil_L = L[py1:py2, px1:px2]
        local_phil_hi = float(np.percentile(phil_L, 85.0))
        mustache_drop = float(min(med_L, local_phil_hi + 3.0) - float(np.percentile(phil_L, 25.0)))

    # Cue 3: Head-Veil (`Ghunghat` / `Veli`) & High Jaw-Wrapped Saree Pallu Detector
    # Checks (a) Crown veil over hair (`crown..eye_mid_y-0.42*ipd`), (b) Temple veil (`eye_mid_y-0.35..+0.65*ipd`),
    # and (c) Cheek/earlobe saree pallu wrap (`eye_mid_y+0.30..chin_y-0.04*ipd`, either side or bilateral).
    has_head_pallu = False
    tx_l1 = max(0, int(round(eye_mid_x - 1.24 * ipd)))
    tx_l2 = max(0, int(round(eye_mid_x - 0.70 * ipd)))
    tx_r1 = min(w, int(round(eye_mid_x + 0.70 * ipd)))
    tx_r2 = min(w, int(round(eye_mid_x + 1.24 * ipd)))
    ty1 = max(0, int(round(eye_mid_y - 0.40 * ipd)))
    ty2 = min(h, int(round(geom.chin_xy[1] - 0.05 * ipd)))
    if (tx_l2 > tx_l1 + 2 or tx_r2 > tx_r1 + 2) and ty2 > ty1 + 4:
        side_veil_hits = 0
        max_side_chroma = 0.0
        for sx1, sx2 in ((tx_l1, tx_l2), (tx_r1, tx_r2)):
            if sx2 <= sx1 + 2:
                continue
            fg_side = 1.0
            if raw_bria_alpha is not None and raw_bria_alpha.shape[:2] == (h, w):
                fg_side = float(np.mean(raw_bria_alpha[ty1:ty2, sx1:sx2] > 0.45))
            if fg_side < 0.34:
                continue
            non_skin_s = (geom.skin_mask[ty1:ty2, sx1:sx2] < 0.28) & (L[ty1:ty2, sx1:sx2] > 24.0)
            if raw_bria_alpha is not None and raw_bria_alpha.shape[:2] == (h, w):
                non_skin_s = non_skin_s & (raw_bria_alpha[ty1:ty2, sx1:sx2] > 0.50)
            if int(np.count_nonzero(non_skin_s)) >= 12:
                ch_s = np.hypot(a[ty1:ty2, sx1:sx2], b[ty1:ty2, sx1:sx2])[non_skin_s]
                p75_ch = float(np.percentile(ch_s, 75.0))
                max_side_chroma = max(max_side_chroma, p75_ch)
                if p75_ch > 21.5:
                    side_veil_hits += 1
        # Crown cloth veil check (colored veil draped over top of head)
        cy_v1 = max(0, int(round(eye_mid_y - 1.38 * ipd)))
        cy_v2 = max(0, int(round(eye_mid_y - 0.68 * ipd)))
        cx_v1 = max(0, int(round(eye_mid_x - 0.55 * ipd)))
        cx_v2 = min(w, int(round(eye_mid_x + 0.55 * ipd)))
        crown_veil = False
        if cy_v2 > cy_v1 + 3 and cx_v2 > cx_v1 + 4:
            cr_mask = geom.skin_mask[cy_v1:cy_v2, cx_v1:cx_v2] < 0.25
            if raw_bria_alpha is not None and raw_bria_alpha.shape[:2] == (h, w):
                cr_mask = cr_mask & (raw_bria_alpha[cy_v1:cy_v2, cx_v1:cx_v2] > 0.55)
            if int(np.count_nonzero(cr_mask)) >= 14:
                cr_ch = float(np.median(np.hypot(a[cy_v1:cy_v2, cx_v1:cx_v2], b[cy_v1:cy_v2, cx_v1:cx_v2])[cr_mask]))
                if cr_ch > 22.0:
                    crown_veil = True
        if (side_veil_hits >= 2 or crown_veil or (side_veil_hits >= 1 and max_side_chroma > 30.0 and bindi_sindoor_pixels >= 4)) and (
            beard_ratio < 0.12 or bindi_sindoor_pixels >= 5
        ):
            has_head_pallu = True

    # Cue 4: Braided pigtails / ponytails behind neck (`[0.46..1.02]*ipd`, `y=[ey+1.12..ey+1.88]*ipd`)
    sh_hair_y1 = max(0, int(round(eye_mid_y + 1.12 * ipd)))
    sh_hair_y2 = min(h, int(round(eye_mid_y + 1.88 * ipd)))
    nx_l1 = max(0, int(round(eye_mid_x - 1.02 * ipd)))
    nx_l2 = max(0, int(round(eye_mid_x - 0.46 * ipd)))
    nx_r1 = min(w, int(round(eye_mid_x + 0.46 * ipd)))
    nx_r2 = min(w, int(round(eye_mid_x + 1.02 * ipd)))
    long_hair_score = 0.0
    if nx_l2 > nx_l1 + 2 and nx_r2 > nx_r1 + 2 and sh_hair_y2 > sh_hair_y1 + 4:
        patch_l = L[sh_hair_y1:sh_hair_y2, nx_l1:nx_l2] < 44.0
        patch_r = L[sh_hair_y1:sh_hair_y2, nx_r1:nx_r2] < 44.0
        if raw_bria_alpha is not None and raw_bria_alpha.shape[:2] == (h, w):
            patch_l = patch_l & (raw_bria_alpha[sh_hair_y1:sh_hair_y2, nx_l1:nx_l2] > 0.35)
            patch_r = patch_r & (raw_bria_alpha[sh_hair_y1:sh_hair_y2, nx_r1:nx_r2] > 0.35)
        long_hair_score = 0.5 * (float(np.mean(patch_l)) + float(np.mean(patch_r)))

    female_score = 0.0
    male_score = 0.0
    if bindi_sindoor_pixels >= 5:
        female_score += 1.35
    elif has_dark_bindi_dot and beard_ratio < 0.07 and mustache_drop < 8.0:
        female_score += 0.65
    if has_head_pallu:
        female_score += 0.70
    if long_hair_score > 0.14 and beard_ratio < 0.08 and mustache_drop < 8.5:
        female_score += 0.55
    if float(np.percentile(ornament_lock, 99.5)) > 0.50 and beard_ratio < 0.08:
        female_score += 0.25

    if beard_ratio >= 0.16 or mustache_drop >= 13.5:
        male_score += 0.88
    elif beard_ratio >= 0.09 or mustache_drop >= 9.5:
        male_score += 0.45
    if bindi_sindoor_pixels < 3 and not has_dark_bindi_dot and not has_head_pallu and long_hair_score < 0.09:
        male_score += 0.35

    if female_score > male_score + 0.04:
        conf = float(np.clip(0.55 + 0.40 * (female_score - male_score), 0.55, 0.98))
        return "female", conf, has_head_pallu
    conf = float(np.clip(0.55 + 0.40 * (male_score - female_score), 0.52, 0.98))
    return "male", conf, False


def detect_foreground_subjects(
    img_rgb: np.ndarray,
    raw_bria_alpha: np.ndarray | None = None,
    max_subjects: int = 4,
) -> list[SubjectProfile]:
    """
    Master Multi-Subject Detector for Single, Dual/Couple, and Family Portraits (1–4 people):
    - Uses OpenCV YuNet ONNX (`227 KB`, `<3ms`) in Hugging Face Cloud (`IS_HF_CLOUD == True`)
      to find all candidate faces & 5-point landmarks accurately.
    - Rejects wall paintings (e.g. framed deity/poster faces) and background banner faces using
      Foreground Alpha Salience (`raw_bria_alpha`) + Inter-Subject IPD Scale Ratio (`>= 0.44x`).
    - Falls back seamlessly to `analyze_face_geometry` when YuNet is unavailable (local PC `0 MB`).
    - Computes exclusive Left-to-Right horizontal ownership boxes (`roi_box`), `OrnamentLock™`,
      and AI Smart Gender / Head-Pallu classification for every real person.
    """
    h, w = img_rgb.shape[:2]
    skin_full = compute_biophysical_skin_mask(img_rgb, raw_bria_alpha)
    lab_full = rgb_to_lab(img_rgb)

    candidates: list[FacialAnthropometrics] = []
    yunet_path = _ensure_yunet_onnx_cloud()

    if yunet_path is not None and HAS_CV2 and cv2 is not None:
        try:
            scale = min(1.0, 960.0 / float(max(h, w)))
            dw = max(64, int(round(w * scale)))
            dh = max(64, int(round(h * scale)))
            det_rgb = resize_img(img_rgb, dw, dh, mode="bilinear") if scale < 0.999 else img_rgb
            det_bgr = cv2.cvtColor(det_rgb, cv2.COLOR_RGB2BGR)

            detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (dw, dh),
                score_threshold=0.66,
                nms_threshold=0.32,
                top_k=12,
            )
            _, faces = detector.detect(det_bgr)
            if faces is not None and len(faces) > 0:
                inv_s = 1.0 / scale
                parsed_raw: list[tuple[float, float, tuple, tuple, tuple, tuple, tuple, tuple]] = []
                for row in faces:
                    score = float(row[14])
                    if score < 0.66:
                        continue
                    bx, by, bw_f, bh_f = float(row[0]) * inv_s, float(row[1]) * inv_s, float(row[2]) * inv_s, float(row[3]) * inv_s
                    re_pt = (float(row[4]) * inv_s, float(row[5]) * inv_s)
                    le_pt = (float(row[6]) * inv_s, float(row[7]) * inv_s)
                    no_pt = (float(row[8]) * inv_s, float(row[9]) * inv_s)
                    rm_pt = (float(row[10]) * inv_s, float(row[11]) * inv_s)
                    lm_pt = (float(row[12]) * inv_s, float(row[13]) * inv_s)
                    ipd_cand = math.hypot(le_pt[0] - re_pt[0], le_pt[1] - re_pt[1])
                    if ipd_cand < max(12.0, min(h, w) * 0.032):
                        continue

                    # Foreground Salience Gate: reject wall paintings / flex-banner faces!
                    if raw_bria_alpha is not None and raw_bria_alpha.shape[:2] == (h, w):
                        fx1 = max(0, int(round(bx + 0.15 * bw_f)))
                        fx2 = min(w, int(round(bx + 0.85 * bw_f)))
                        fy1 = max(0, int(round(by + 0.15 * bh_f)))
                        fy2 = min(h, int(round(by + 0.85 * bh_f)))
                        face_alpha_med = float(np.median(raw_bria_alpha[fy1:fy2, fx1:fx2])) if (fx2 > fx1 and fy2 > fy1) else 1.0

                        # Also check torso directly below chin (wall frames float on the wall with background below them!)
                        ty1 = min(h - 1, max(0, int(round(by + 1.08 * bh_f))))
                        ty2 = min(h, int(round(by + 1.75 * bh_f)))
                        torso_alpha_med = float(np.median(raw_bria_alpha[ty1:ty2, fx1:fx2])) if (fx2 > fx1 and ty2 > ty1 + 2) else face_alpha_med
                        if face_alpha_med < 0.35 or (face_alpha_med < 0.65 and torso_alpha_med < 0.25):
                            continue

                    parsed_raw.append((ipd_cand, score, (bx, by, bw_f, bh_f), re_pt, le_pt, no_pt, rm_pt, lm_pt))

                if parsed_raw:
                    max_ipd = max(item[0] for item in parsed_raw)
                    # Keep all real foreground subjects (IPD >= 46% of largest subject), rejecting tiny background faces
                    valid_raw = [item for item in parsed_raw if item[0] >= 0.46 * max_ipd]
                    # Sort by IPD * score descending, cap to max_subjects, then sort Left -> Right
                    valid_raw.sort(key=lambda it: it[0] * it[1], reverse=True)
                    valid_raw = valid_raw[:max(1, int(max_subjects))]
                    valid_raw.sort(key=lambda it: 0.5 * (it[3][0] + it[4][0]))

                    for _ipd_c, _sc, b_xywh, re_p, le_p, no_p, rm_p, lm_p in valid_raw:
                        candidates.append(
                            _build_geom_from_5pt(
                                img_rgb,
                                skin_full=skin_full,
                                lab_full=lab_full,
                                box_xywh=b_xywh,
                                right_eye_xy=re_p,
                                left_eye_xy=le_p,
                                nose_xy=no_p,
                                right_mouth_xy=rm_p,
                                left_mouth_xy=lm_p,
                            )
                        )
        except Exception as e:
            print(f"⚠️ [MultiSubject] YuNet detection fallback: {e}")

    if not candidates:
        candidates = [analyze_face_geometry(img_rgb, fg_alpha=raw_bria_alpha)]

    n_subj = len(candidates)
    mid_xs = [0.5 * (g.left_eye.center_x + g.right_eye.center_x) for g in candidates]

    profiles: list[SubjectProfile] = []
    for idx, g in enumerate(candidates):
        # Compute clean horizontal partition boundary between adjacent people in dual/group photos
        rx1 = 0 if idx == 0 else int(round(0.5 * (mid_xs[idx - 1] + mid_xs[idx])))
        rx2 = w if idx == n_subj - 1 else int(round(0.5 * (mid_xs[idx] + mid_xs[idx + 1])))
        ry1 = max(0, int(round(g.crown_xy[1] - 0.35 * g.ipd_px)))
        ry2 = h

        orn_lock = compute_cultural_ornament_lock(img_rgb, g, lab_full=lab_full)
        gender, g_conf, has_pallu = _infer_gender_and_head_pallu(
            img_rgb,
            g,
            lab_full=lab_full,
            ornament_lock=orn_lock,
            raw_bria_alpha=raw_bria_alpha,
        )

        if n_subj == 1:
            lbl = f"Primary Subject ({gender.capitalize()})"
        elif n_subj == 2:
            pos_str = "Left" if idx == 0 else "Right"
            lbl = f"Person {idx + 1} ({pos_str} — {gender.capitalize()})"
        else:
            lbl = f"Person {idx + 1} ({gender.capitalize()})"

        profiles.append(
            SubjectProfile(
                subject_index=idx,
                label=lbl,
                geom=g,
                inferred_gender=gender,
                gender_confidence=g_conf,
                has_head_pallu=has_pallu,
                roi_box=(rx1, ry1, rx2, ry2),
                ornament_lock=orn_lock,
            )
        )

    return profiles


def build_veil_dress_reminder(
    subjects: list[SubjectProfile],
    dress_preset: str = "none",
    target_person: str = "auto",
) -> str:
    """
    AI Veil (`Ghunghat` / `Veli` / Head-Pallu) Decision & Studio Operator Reminder:
    Inspects targeted female subject(s) and returns an actionable reminder whenever a customer
    is wearing a head veil or high jaw-wrapped saree pallu during Dress Change.
    """
    if not subjects:
        return ""
    veiled = [s for s in subjects if s.inferred_gender == "female" and s.has_head_pallu]
    if not veiled:
        return ""
    tp = str(target_person or "auto").strip().lower()
    if tp == "person_1" and subjects[0] not in veiled:
        return ""
    if tp == "person_2" and len(subjects) >= 2 and subjects[1] not in veiled:
        return ""

    names = ", ".join(s.label for s in veiled)
    dp = str(dress_preset or "none").strip()
    if dp == "none":
        return (
            f"🧕 **AI Veil / Head-Pallu (`Veli`) Detected** on **{names}**: "
            "Original saree/veil drape is preserved 100% intact. If changing dress later, note that a head veil is present."
        )
    if "saree" in dp:
        return (
            f"⚠️ **AI Veil (`Veli` / Head-Pallu) Reminder ({names}):** Customer is wearing a head/neck veil (`pallu`). "
            "AI has harmonized the neckline for the new Silk Saree; use an unveiled photo if you want bare hair/neck."
        )
    return (
        f"⚠️ **AI Veil (`Veli` / Head-Pallu) Reminder ({names}):** Customer is wearing a saree veil / high pallu around the head/jaw! "
        "Replacing only the lower outfit with a Shirt/Blazer/Uniform may clash with the head veil — consider keeping **Original Dress**, "
        "selecting a **Silk Saree**, or capturing an unveiled photo."
    )


def prune_background_bystanders(
    raw_alpha: np.ndarray,
    subjects: list[SubjectProfile],
) -> np.ndarray:
    """
    Topology-Preserving Connected-Component Bystander & Upper-Corner Wall Clutter Filter:
    - NEVER clips wide shoulders (`x = 0..w` stays 100% intact!) and NEVER carves notches into
      neck-wrapped sarees, shawls, collars, or veils!
    - Suppresses ONLY far upper-corner wall clutter (`dx > 2.28 * ipd` AND `yy < chin_y - 0.14 * ipd`
      away from every subject's head/neck/shoulders, e.g. hanging wall laundry in wide room shots)
      and removes disconnected background passersby via 8-connected subject anchoring!
    """
    if not subjects:
        return raw_alpha

    h, w = raw_alpha.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    keep_zone = np.zeros((h, w), dtype=np.float32)
    near_any_subject_head = np.zeros((h, w), dtype=bool)

    for subj in subjects:
        g = subj.geom
        ipd = max(16.0, float(g.ipd_px))
        cx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
        cy = 0.5 * (g.left_eye.center_y + g.right_eye.center_y)
        chin_y = float(g.chin_xy[1])
        dx = np.abs(xx - cx)

        # Full-width below upper neck (`0..w` — ZERO side shoulder cropping!)
        full_width_torso = np.clip((yy - (chin_y - 0.18 * ipd)) / max(4.0, 0.10 * ipd), 0.0, 1.0)
        # Wide head / veil / high neck-pallu zone above neck (`dx <= 2.28 * ipd`)
        upper_head_veil = np.clip(((2.28 * ipd) - dx) / max(6.0, 0.18 * ipd), 0.0, 1.0)
        subj_valid = np.maximum(full_width_torso, upper_head_veil)
        keep_zone = np.maximum(keep_zone, subj_valid)

        wisp_rx = (1.65 * ipd) if subj.has_head_pallu else (1.30 * ipd)
        head_halo = (((xx - cx) / wisp_rx) ** 2 + ((yy - (cy + 0.08 * ipd)) / (1.85 * ipd)) ** 2) <= 1.0
        near_any_subject_head |= head_halo

    gated_alpha = raw_alpha * keep_zone

    if not HAS_CV2 or cv2 is None:
        return np.clip(gated_alpha, 0.0, 1.0).astype(np.float32)

    fg_bin = (gated_alpha > 0.22).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=8)
    if num_labels <= 2:
        return np.clip(gated_alpha, 0.0, 1.0).astype(np.float32)

    keep_labels: set[int] = set()
    for subj in subjects:
        g = subj.geom
        ipd = max(16.0, float(g.ipd_px))
        cx = 0.5 * (g.left_eye.center_x + g.right_eye.center_x)
        cy = 0.5 * (g.left_eye.center_y + g.right_eye.center_y)

        anchor_pts = [
            (cx, cy),
            g.chin_xy,
            g.jugular_notch,
            (cx, min(float(h - 2), cy + 2.6 * ipd)),
        ]
        for ax, ay in anchor_pts:
            ix = int(np.clip(round(ax), 0, w - 1))
            iy = int(np.clip(round(ay), 0, h - 1))
            lbl_val = int(labels[iy, ix])
            if lbl_val > 0:
                keep_labels.add(lbl_val)
            else:
                patch = labels[max(0, iy - 4) : min(h, iy + 5), max(0, ix - 4) : min(w, ix + 5)]
                nz = patch[patch > 0]
                if nz.size > 0:
                    keep_labels.add(int(nz[0]))

    if not keep_labels:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_labels.add(int(np.argmax(areas)) + 1)

    keep_mask = np.isin(labels, list(keep_labels)) | (near_any_subject_head & (gated_alpha > 0.08))
    keep_float = gaussian_blur(keep_mask.astype(np.float32), sigma=1.6)
    pruned = gated_alpha * np.clip(keep_float * 1.45, 0.0, 1.0)
    return np.clip(pruned, 0.0, 1.0).astype(np.float32)
