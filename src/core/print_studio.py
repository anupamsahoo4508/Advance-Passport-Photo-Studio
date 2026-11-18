"""
ICAO 9303 Biometric Auto-Alignment, Smart Framing, Name/Date Plaque & 4K Print Sheet Studio.
Supports:
- SO(2) Inter-Pupillary Horizon Tilt Correction
- ICAO 9303 Crown-to-Chin & Eye-Line Biometric Auto-Cropping
- Exam Name & Date Stamp Plaque (SSC / UPSC / Banking / Visa standard)
- 300/600 DPI Studio Print Layouts (4x6 inch 8-up/12-up, A4 30-up/36-up with cutting guides)
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.shared.config import PASSPORT_STANDARDS
from src.core.cv_ops import resize_img, rotate_image
from src.core.facial_math_engine import FacialAnthropometrics, analyze_face_geometry


PRINT_SHEET_LAYOUTS = {
    "single_photo": {
        "label": "🖼️ Single Photo Only",
        "canvas_px": None,
        "grid": (1, 1),
    },
    "sheet_4x6_8up": {
        "label": "🖨️ 4×6 Sheet — 8 Photos (4×2)",
        "canvas_px": (1800, 1200),  # 6x4 inch landscape at 300 DPI
        "grid": (4, 2),
        "photo_px": (390, 500),
    },
    "sheet_4x6_12up": {
        "label": "🖨️ 4×6 Sheet — 12 Photos (4×3)",
        "canvas_px": (1800, 1200),
        "grid": (4, 3),
        "photo_px": (350, 340),
    },
    "sheet_a4_30up": {
        "label": "📄 A4 Sheet — 30 Photos (6×5)",
        "canvas_px": (2480, 3508),  # A4 portrait at 300 DPI
        "grid": (6, 5),
        "photo_px": (360, 462),
    },
    "sheet_a4_36up": {
        "label": "📄 A4 Sheet — 36 Photos (6×6)",
        "canvas_px": (2480, 3508),
        "grid": (6, 6),
        "photo_px": (355, 455),
    },
}


def auto_level_portrait(
    img_rgb: np.ndarray,
    geom: FacialAnthropometrics | None = None,
    bg_rgb: tuple[int, int, int] | None = None,
    max_correction_deg: float = 15.0,
) -> np.ndarray:
    """
    Level head tilt (SO(2) roll angle) so the inter-pupillary line is strictly horizontal.
    When bg_rgb is None, estimates corner border fill from the image's own top corners so
    subsequent foreground alpha matting corner statistics remain uncorrupted.
    """
    if geom is None:
        geom = analyze_face_geometry(img_rgb)
    if abs(geom.roll_deg) < 0.45 or abs(geom.roll_deg) > max_correction_deg:
        return img_rgb.copy()

    if bg_rgb is None:
        h, w = img_rgb.shape[:2]
        pad = max(2, int(min(h, w) * 0.05))
        corners = np.concatenate([
            img_rgb[:pad, :pad, :3].reshape(-1, 3),
            img_rgb[:pad, max(0, w - pad):, :3].reshape(-1, 3),
        ], axis=0)
        med = np.median(corners, axis=0)
        fill_rgb = (int(round(med[0])), int(round(med[1])), int(round(med[2])))
    else:
        fill_rgb = bg_rgb

    # Rotate by roll_deg to bring eye horizon horizontal
    return rotate_image(img_rgb, angle_deg=geom.roll_deg, border_rgb=fill_rgb)


def icao_biometric_crop(
    img_rgb: np.ndarray,
    standard_key: str = "india_passport_35x45mm",
    geom: FacialAnthropometrics | None = None,
    bg_rgb: tuple[int, int, int] = (152, 243, 255),
) -> np.ndarray:
    """
    Perform ICAO 9303 Biometric Framing:
    Positions the crown-to-chin head height at ~62% of frame height and centers the
    inter-pupillary midpoint horizontally with the eye line at ~61% from bottom edge.
    Preserves current pixel density (4K-ready) and extends gradient/solid backdrops seamlessly.
    """
    std = PASSPORT_STANDARDS.get(standard_key, PASSPORT_STANDARDS["india_passport_35x45mm"])
    if std["aspect_w"] is None or std["aspect_h"] is None:
        return img_rgb.copy()

    h, w = img_rgb.shape[:2]
    if geom is None or (geom.img_w, geom.img_h) != (w, h):
        geom = analyze_face_geometry(img_rgb)

    aspect = float(std["aspect_w"]) / float(std["aspect_h"])
    head_h = max(32.0, geom.chin_xy[1] - geom.crown_xy[1])
    target_head_ratio = float(std.get("head_ratio_target", 0.62))
    eye_from_bottom = float(std.get("eye_line_from_bottom", 0.61))

    crop_h = head_h / target_head_ratio
    crop_w = crop_h * aspect

    # Clamp crop dimensions to reasonable bounds relative to source image
    crop_h = max(h * 0.45, min(h * 1.18, crop_h))
    crop_w = crop_h * aspect
    if crop_w > w * 1.18:
        crop_w = w * 1.18
        crop_h = crop_w / aspect

    eye_mid_x = 0.5 * (geom.left_eye.center_x + geom.right_eye.center_x)
    eye_mid_y = 0.5 * (geom.left_eye.center_y + geom.right_eye.center_y)

    # Eye line sits at (1 - eye_from_bottom) from top of crop
    x1 = int(round(eye_mid_x - 0.5 * crop_w))
    y1 = int(round(eye_mid_y - (1.0 - eye_from_bottom) * crop_h))
    out_w = max(64, int(round(crop_w)))
    out_h = max(64, int(round(crop_h)))
    x2 = x1 + out_w
    y2 = y1 + out_h

    # If crop extends above y=0, check whether the subject's hair is already clipped by the top edge y=0.
    # If hair touches y=0, shift the crop box down to y1=0 so the hair stays naturally flush with the top border
    # (just like the original camera photo) instead of synthesizing an artificial patch above the head.
    work_img = img_rgb
    if y1 < 0:
        pad_c = max(2, min(16, w // 10))
        c_left = np.median(img_rgb[:pad_c, :pad_c, :3], axis=(0, 1)).astype(np.float32)
        c_right = np.median(img_rgb[:pad_c, max(0, w - pad_c):, :3], axis=(0, 1)).astype(np.float32)
        t_x = np.linspace(0.0, 1.0, w, dtype=np.float32)[:, None]
        bg_ref_line = (1.0 - t_x) * c_left[None, :] + t_x * c_right[None, :]
        top_row = img_rgb[0, :, :3].astype(np.float32)
        diff_bg = np.sqrt(np.sum((top_row - bg_ref_line) ** 2, axis=-1))
        fg_cols = np.where(diff_bg > 18.0)[0]
        if len(fg_cols) >= 4:
            y1 = 0
            y2 = y1 + out_h
        elif len(fg_cols) > 0:
            # 1-3 stray outlier pixels on row y=0: sanitize them to bg_ref_line so edge-padding never creates vertical streaks
            work_img = img_rgb.copy()
            work_img[0, fg_cols, :3] = np.clip(np.round(bg_ref_line[fg_cols]), 0, 255).astype(np.uint8)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = np.pad(
            work_img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        sx1 = x1 + pad_left
        sy1 = y1 + pad_top
        return padded[sy1:sy1 + out_h, sx1:sx1 + out_w].copy()

    return work_img[y1:y1 + out_h, x1:x1 + out_w].copy()


def apply_name_date_plaque(
    img_rgb: np.ndarray,
    candidate_name: str = "",
    dob_or_date: str = "",
    add_border: bool = True,
) -> np.ndarray:
    """
    Add optional official black/white border and exam Name + Date of Photo plaque at bottom.
    """
    h, w = img_rgb.shape[:2]
    im = Image.fromarray(img_rgb, mode="RGB")
    draw = ImageDraw.Draw(im)

    name_clean = (candidate_name or "").strip()
    date_clean = (dob_or_date or "").strip()

    if name_clean or date_clean:
        plaque_h = max(28, int(round(h * 0.135)))
        y_top = h - plaque_h
        draw.rectangle([0, y_top, w, h], fill=(255, 255, 255), outline=(20, 20, 20), width=max(1, w // 250))

        lines = [x for x in (name_clean.upper(), date_clean) if x]
        font_size = max(11, int(plaque_h / (1.8 if len(lines) == 1 else 2.5)))
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for idx, text in enumerate(lines):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = max(4, (w - tw) // 2)
            if len(lines) == 1:
                ty = y_top + (plaque_h - th) // 2 - 2
            else:
                ty = y_top + int((idx + 0.18) * (plaque_h / 2.0))
            draw.text((tx, ty), text, fill=(10, 10, 10), font=font)

    if add_border:
        bw = max(2, int(round(min(h, w) * 0.008)))
        draw.rectangle([0, 0, w - 1, h - 1], outline=(24, 24, 24), width=bw)

    return np.array(im, dtype=np.uint8)


def generate_print_sheet(
    single_passport_rgb: np.ndarray,
    layout_key: str = "sheet_4x6_8up",
    draw_cut_marks: bool = True,
) -> np.ndarray:
    """
    Generate a 300 DPI studio print sheet (4x6 inch or A4) populated with the passport photo.
    """
    layout = PRINT_SHEET_LAYOUTS.get(layout_key, PRINT_SHEET_LAYOUTS["single_photo"])
    if layout["canvas_px"] is None:
        return single_passport_rgb.copy()

    sheet_w, sheet_h = layout["canvas_px"]
    cols, rows = layout["grid"]
    photo_w, photo_h = layout["photo_px"]

    tile = resize_img(single_passport_rgb, photo_w, photo_h, mode="lanczos")
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    # Calculate even margins & gutters
    total_tiles_w = cols * photo_w
    total_tiles_h = rows * photo_h
    gap_x = max(12, (sheet_w - total_tiles_w) // (cols + 1))
    gap_y = max(12, (sheet_h - total_tiles_h) // (rows + 1))

    start_x = max(8, (sheet_w - (cols * photo_w + (cols - 1) * gap_x)) // 2)
    start_y = max(8, (sheet_h - (rows * photo_h + (rows - 1) * gap_y)) // 2)

    tile_im = Image.fromarray(tile, mode="RGB")

    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (photo_w + gap_x)
            y = start_y + r * (photo_h + gap_y)
            sheet.paste(tile_im, (x, y))
            # Thin border around each tile for scissor cutting precision
            draw.rectangle([x - 1, y - 1, x + photo_w, y + photo_h], outline=(170, 175, 185), width=1)
            if draw_cut_marks:
                mk = 8
                for cx_m, cy_m in ((x, y), (x + photo_w, y), (x, y + photo_h), (x + photo_w, y + photo_h)):
                    draw.line([(cx_m - mk, cy_m), (cx_m + mk, cy_m)], fill=(120, 120, 120), width=1)
                    draw.line([(cx_m, cy_m - mk), (cx_m, cy_m + mk)], fill=(120, 120, 120), width=1)

    return np.array(sheet, dtype=np.uint8)
