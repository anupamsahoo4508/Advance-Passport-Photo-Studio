"""
Backup Manager — Archive creation, storage monitoring, and purge.
"""
from __future__ import annotations
import os
import shutil
import time
import json
import sqlite3
import io
import zipfile
from datetime import datetime
from src.shared.config import BASE_DATA_DIR, DB_DIR, BACKUP_DIR, DB_PATH, MAX_STORAGE_MB

STATUS_FILE = os.path.join(BASE_DATA_DIR, "system_status.json")


def get_folder_size_mb(folder: str) -> float:
    """Calculate total size of a folder in megabytes."""
    total_size = 0
    if not os.path.exists(folder):
        return 0.0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def get_system_status() -> tuple[float, float, str, float, str]:
    """Get storage usage, backup status, and date range of stored data."""
    size_mb = get_folder_size_mb(DB_DIR)
    percentage = min(100.0, (size_mb / MAX_STORAGE_MB) * 100.0)

    last_backup_str = "Never"
    days_passed = 0.0
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                last_time = data.get("last_backup", time.time())
                days_passed = (time.time() - last_time) / 86400.0
                last_backup_str = f"{int(days_passed)} days ago"
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    date_range = "No Data Available"
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM customers")
            mn, mx = cursor.fetchone()
            conn.close()
            if mn and mx:
                d1 = datetime.fromtimestamp(mn).strftime('%d-%b-%Y')
                d2 = datetime.fromtimestamp(mx).strftime('%d-%b-%Y')
                date_range = f"[{d1} To {d2}]"
        except sqlite3.OperationalError:
            pass

    return size_mb, percentage, last_backup_str, days_passed, date_range


def create_archive_package() -> tuple[io.BytesIO | None, str | None]:
    """Create a zip backup of the database directory in-memory."""
    _, _, _, _, range_str = get_system_status()
    clean_range = range_str.replace("[", "").replace("]", "").replace(" ", "_") if "To" in range_str else "Empty"
    backup_name = f"Maa_Sitala_Backup_{clean_range}_{int(time.time())}.zip"

    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        return None, None

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DB_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(file_path, os.path.relpath(file_path, DB_DIR))

    memory_file.seek(0)
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_backup": time.time()}, f)

    return memory_file, backup_name


def create_backup_file_on_disk() -> str | None:
    """Write a ZIP backup archive to BACKUP_DIR and return its file path (for Gradio DownloadButton/File)."""
    mem, name = create_archive_package()
    if mem is None or name is None:
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(BACKUP_DIR, name))
    with open(out_path, "wb") as f:
        f.write(mem.getvalue())
    return out_path


def purge_system() -> str:
    """Clear all database data and reinitialize."""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    from src.shared.cloud_database import init_db
    init_db()
    return "🗑️ Storage cleared and reset to 0.0 MB."
