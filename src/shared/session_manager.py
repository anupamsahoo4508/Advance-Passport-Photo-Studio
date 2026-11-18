"""
Session Manager — Temporary session history with auto-cleanup.
"""
from __future__ import annotations
import os
import time
import json
import glob
import shutil
import numpy as np
from src.shared.config import HISTORY_DIR
from src.core.cv_ops import imwrite_rgb


def cleanup_old_sessions(days: int = 3):
    """Remove sessions older than specified days."""
    current_time = time.time()
    expire_time = days * 86400
    for session_folder in glob.glob(os.path.join(HISTORY_DIR, "*")):
        if os.path.isdir(session_folder):
            if (current_time - os.path.getmtime(session_folder)) > expire_time:
                shutil.rmtree(session_folder, ignore_errors=True)


def save_session(input_img: np.ndarray, final_img_path: str, hex_color: str, upscale_val: int) -> str:
    """Save a processing session for history."""
    cleanup_old_sessions()
    os.makedirs(HISTORY_DIR, exist_ok=True)
    existing_count = len([f for f in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, f))])
    session_id = str(existing_count + 1)

    session_path = os.path.join(HISTORY_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    input_save_path = os.path.join(session_path, "input.jpg")
    imwrite_rgb(input_save_path, input_img[:, :, :3], quality=95)

    final_save_path = os.path.join(session_path, "final.png")
    if os.path.exists(final_img_path):
        shutil.copy(final_img_path, final_save_path)

    meta_data = {"color": hex_color, "upscale": upscale_val, "timestamp": time.time()}
    with open(os.path.join(session_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    return final_save_path


def get_recent_gallery() -> list[tuple[str, str]]:
    """Get list of recent processed images for gallery view."""
    if not os.path.exists(HISTORY_DIR):
        return []
    sessions = sorted(glob.glob(os.path.join(HISTORY_DIR, "*")), key=os.path.getmtime, reverse=True)
    gallery = []
    for s in sessions:
        final_img = os.path.join(s, "final.png")
        if os.path.exists(final_img):
            gallery.append((final_img, os.path.basename(s)))
    return gallery
