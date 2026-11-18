"""
Cloud Database — SQLite local engine + Hugging Face Dataset sync (DevAnupam2008/Studio-Database).
Security: HF_TOKEN is read ONLY from environment/vault, never hardcoded.
Supports V2 + V3 Bio-Optical & SartorialFit Dress recipes with zero-cv2-lock-in Pillow I/O.
"""
from __future__ import annotations
import sqlite3
import os
import time
import json
import threading
from datetime import datetime
import numpy as np
from src.shared.config import HF_TOKEN, REPO_ID, DB_DIR, DB_PATH, IMG_DIR, THUMB_DIR, IS_SHOWCASE_MODE
from src.core.cv_ops import imread_rgb, imwrite_rgb, resize_img

try:
    from huggingface_hub import HfApi, login  # type: ignore
    HAS_HF_HUB = True
except ImportError:
    HfApi = None
    login = None
    HAS_HF_HUB = False

# === Cloud Authentication ===
api = None
if HAS_HF_HUB and HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        api = HfApi(token=HF_TOKEN)
    except Exception as _e:
        print(f"⚠️ Hugging Face Hub login skipped: {_e}")


def pull_from_cloud() -> bool:
    """Download database from HF Dataset on startup if HF_TOKEN and huggingface_hub are available."""
    if IS_SHOWCASE_MODE or not REPO_ID:
        init_db()
        print("ℹ️ Running in Public Showcase Preview Mode (No Dataset sync required).")
        return False
    if not HAS_HF_HUB or not HF_TOKEN or not api:
        print("ℹ️ Running in Local SQLite Mode (Cloud sync ready when deployed to HF Spaces).")
        return False
    try:
        print(f"☁️ Pulling database from {REPO_ID}...")
        api.snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=DB_DIR)
        init_db()
        print("✅ Cloud Sync Complete!")
        return True
    except Exception as e:
        print(f"⚠️ Cloud Pull Skipped/Failed: {e}")
        return False


def push_to_cloud() -> bool:
    """Upload database folder to HF Dataset in background thread with retry."""
    if IS_SHOWCASE_MODE or not REPO_ID or not HAS_HF_HUB or not HF_TOKEN or not api:
        return False
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=DB_DIR,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Auto-Sync: Customer Update ({int(time.time())})",
            )
            print("☁️ Data pushed to Hugging Face Dataset.")
            return True
        except Exception as e:
            print(f"⚠️ Cloud Push Attempt {attempt + 1}/3 Failed: {e}")
            time.sleep(2)
    return False


def init_db():
    """Ensure SQLite database, customer image/thumbnail folders, and customer_shares table exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)
    os.makedirs(os.path.join(DB_DIR, "shares"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cust_id TEXT UNIQUE,
            name TEXT,
            input_image_path TEXT,
            thumbnail_path TEXT,
            recipe_json TEXT,
            timestamp REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            share_id TEXT UNIQUE,
            customer_name TEXT,
            pin_salt TEXT,
            pin_sha256 TEXT,
            pin_hash TEXT,
            photo_path TEXT,
            sheet_path TEXT,
            resolution_str TEXT,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            created_at REAL
        )
    ''')
    cursor.execute("PRAGMA table_info(customer_shares)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col_name, col_type in (("pin_sha256", "TEXT"), ("width", "INTEGER DEFAULT 0"), ("height", "INTEGER DEFAULT 0")):
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE customer_shares ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass
    conn.commit()
    conn.close()


init_db()


def get_next_ms_id() -> str:
    """Generate the next unique customer ID (ms-1, ms-2...). Never reuses deleted IDs."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(CAST(SUBSTR(cust_id, 4) AS INTEGER)) FROM customers")
    row = cursor.fetchone()
    max_num = (row[0] if row and row[0] is not None else 0)
    conn.close()
    return f"ms-{int(max_num) + 1}"


def check_existing_customer(name: str) -> list:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT cust_id, name, thumbnail_path FROM customers WHERE name = ?', (name,))
    result = cursor.fetchall()
    conn.close()
    return result


def save_customer_recipe(
    name: str,
    input_img: np.ndarray,
    custom_color: str = "#98f3ff",
    upscale_factor: int = 4,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    update_cust_id: str | None = None,
    extra_recipe: dict | None = None,
) -> tuple[str, str | None]:
    """Save or update a customer record, high-res source image, thumbnail, and V3 recipe JSON."""
    if input_img is None:
        return "❌ No Image Detected", None

    name_clean = (name or "").strip() or "Walk-in Customer"
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    recipe_dict = {
        "color": custom_color,
        "upscale": int(upscale_factor),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "gamma": float(gamma),
    }
    if extra_recipe:
        recipe_dict.update(extra_recipe)
    recipe_json = json.dumps(recipe_dict)

    # Generate 120px low-res thumbnail
    h, w = input_img.shape[:2]
    scale = 120.0 / max(1, max(h, w))
    thumb_img = resize_img(input_img[:, :, :3], max(1, int(w * scale)), max(1, int(h * scale)), mode="area")

    if update_cust_id and update_cust_id.strip():
        cust_id = update_cust_id.strip()
        cursor.execute('SELECT input_image_path, thumbnail_path FROM customers WHERE cust_id = ?', (cust_id,))
        paths = cursor.fetchone()
        if paths:
            for p in paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        input_save_path = os.path.join(IMG_DIR, f"{cust_id}_input.jpg")
        thumb_save_path = os.path.join(THUMB_DIR, f"{cust_id}_thumb.jpg")
        imwrite_rgb(input_save_path, input_img[:, :, :3], quality=100)
        imwrite_rgb(thumb_save_path, thumb_img, quality=95)

        if not os.path.exists(input_save_path) or os.path.getsize(input_save_path) == 0:
            conn.close()
            return "❌ Image save failed — disk error", None

        cursor.execute('''
            UPDATE customers SET name=?, input_image_path=?, thumbnail_path=?, recipe_json=?, timestamp=?
            WHERE cust_id=?
        ''', (name_clean, input_save_path, thumb_save_path, recipe_json, time.time(), cust_id))
    else:
        existing = check_existing_customer(name_clean)
        final_name = f"{name_clean} ({len(existing)})" if existing else name_clean

        cust_id = get_next_ms_id()
        input_save_path = os.path.join(IMG_DIR, f"{cust_id}_input.jpg")
        thumb_save_path = os.path.join(THUMB_DIR, f"{cust_id}_thumb.jpg")
        imwrite_rgb(input_save_path, input_img[:, :, :3], quality=100)
        imwrite_rgb(thumb_save_path, thumb_img, quality=95)

        if not os.path.exists(input_save_path) or os.path.getsize(input_save_path) == 0:
            conn.close()
            return "❌ Image save failed — disk error", None

        cursor.execute('''
            INSERT INTO customers (cust_id, name, input_image_path, thumbnail_path, recipe_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (cust_id, final_name, input_save_path, thumb_save_path, recipe_json, time.time()))

    conn.commit()
    if cursor.rowcount == 0:
        conn.close()
        return "❌ Database commit failed", None

    conn.close()
    threading.Thread(target=push_to_cloud, daemon=True).start()
    return f"✅ Saved! ID: {cust_id}", cust_id


def search_database(query: str = "") -> list[tuple[str, str]]:
    """Search customers by ID ('ms-5' or '5') or case-insensitive name substring."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT cust_id, name, thumbnail_path FROM customers ORDER BY timestamp DESC')
    all_data = cursor.fetchall()
    conn.close()

    if not all_data:
        return []

    if not query or not query.strip():
        return [(row[2], f"{row[0]}: {row[1]}") for row in all_data[:50]]

    query_lower = query.strip().lower()

    if query_lower.startswith("ms-"):
        results = [row for row in all_data if row[0].lower() == query_lower]
        if results:
            return [(row[2], f"{row[0]}: {row[1]}") for row in results]

    if query_lower.isdigit():
        results = [row for row in all_data if "-" in row[0] and row[0].split("-")[1].startswith(query_lower)]
        if results:
            return [(row[2], f"{row[0]}: {row[1]}") for row in results]

    results = [row for row in all_data if query_lower in row[1].lower()]
    if results:
        return [(row[2], f"{row[0]}: {row[1]}") for row in results]

    results = [row for row in all_data if any(w.lower().startswith(query_lower) for w in row[1].split())]
    return [(row[2], f"{row[0]}: {row[1]}") for row in results]


def search_customers_detailed(query: str = "") -> list[dict]:
    """
    Fast, rich customer search returning structured records with ID, Name, Formatted Date,
    Recipe Summary, and Thumbnail Path for the Saved Customer studio window.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT cust_id, name, thumbnail_path, recipe_json, timestamp FROM customers ORDER BY timestamp DESC'
    )
    all_data = cursor.fetchall()
    conn.close()

    if not all_data:
        return []

    q = (query or "").strip().lower()
    if not q:
        matched = all_data[:100]
    elif q.startswith("ms-"):
        matched = [r for r in all_data if r[0].lower() == q]
        if not matched:
            matched = [r for r in all_data if q in r[0].lower() or q in r[1].lower()]
    elif q.isdigit():
        matched = [r for r in all_data if "-" in r[0] and r[0].split("-")[1].startswith(q)]
        if not matched:
            matched = [r for r in all_data if q in r[1].lower()]
    else:
        matched = [r for r in all_data if q in r[1].lower() or q in r[0].lower()]
        if not matched:
            matched = [r for r in all_data if any(w.lower().startswith(q) for w in r[1].split())]

    detailed: list[dict] = []
    for cid, cname, thumb_path, rjson, ts in matched:
        date_str = datetime.fromtimestamp(ts).strftime('%d-%b-%Y %I:%M %p') if ts else "Recent"
        try:
            rcp = json.loads(rjson) if rjson else {}
        except Exception:
            rcp = {}
        up_tier = rcp.get("upscale", 4)
        bg_col = rcp.get("color", "#98f3ff")
        dress_key = rcp.get("dress_preset", "none")
        dress_short = "Original Dress" if dress_key == "none" else dress_key.replace("_", " ").title()
        summary = f"Upscale {up_tier}/10 • BG {bg_col} • {dress_short}"
        detailed.append({
            "cust_id": cid,
            "name": cname,
            "thumbnail_path": thumb_path,
            "date": date_str,
            "summary": summary,
        })
    return detailed


def get_all_gallery() -> list[dict]:
    """Return all customers sorted by date (recent first) with formatted dates."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT cust_id, name, thumbnail_path, timestamp FROM customers ORDER BY timestamp DESC')
    all_data = cursor.fetchall()
    conn.close()

    gallery = []
    for row in all_data:
        date_str = datetime.fromtimestamp(row[3]).strftime('%d-%b-%Y %I:%M %p') if row[3] else "Unknown"
        gallery.append({
            "cust_id": row[0],
            "name": row[1],
            "thumbnail_path": row[2],
            "thumbnail_url": f"/api/thumbnail/{row[0]}",
            "date": date_str,
        })
    return gallery


def delete_customer(cust_id: str) -> str:
    """Delete a customer and their image files without affecting other customer IDs."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT input_image_path, thumbnail_path FROM customers WHERE cust_id = ?', (cust_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return f"❌ Customer {cust_id} not found"

    for path in row:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    cursor.execute('DELETE FROM customers WHERE cust_id = ?', (cust_id,))
    conn.commit()
    conn.close()
    threading.Thread(target=push_to_cloud, daemon=True).start()
    return f"✅ {cust_id} deleted successfully"


def load_permanent_customer_recipe(cust_id: str) -> tuple[np.ndarray | None, dict | None]:
    """Load a customer's original RGB image and saved recipe dictionary by cust_id."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, input_image_path, recipe_json FROM customers WHERE cust_id = ?', (cust_id,))
    row = cursor.fetchone()
    conn.close()

    if row and row[1] and os.path.exists(row[1]):
        input_img = imread_rgb(row[1])
        recipe = json.loads(row[2]) if row[2] else {}
        recipe["name"] = row[0]
        recipe["cust_id"] = cust_id
        return input_img, recipe
    return None, None
