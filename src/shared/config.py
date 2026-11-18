"""
Centralized configuration — all constants, biometric standards, and environment variables.
Security: HF_TOKEN is read ONLY from environment or central vault, never hardcoded in source.
"""
import hashlib
import os


_VAULT_PATHS = (
    r"D:\antigravity-core\antigravity-personal-core-hub\.env",
    r"C:\Users\star\.gemini\antigravity\scratch\antigravity-personal-core-hub\.env",
)


def _read_vault_key(keys: tuple[str, ...]) -> str | None:
    """Read the first matching key from the central multi-account `.env` vault."""
    for vault_path in _VAULT_PATHS:
        if os.path.exists(vault_path):
            try:
                with open(vault_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() in keys:
                                val = v.strip().strip('"').strip("'")
                                if val:
                                    return val
            except OSError:
                pass
    return None


def _load_hf_token(account: str = "professional") -> str | None:
    """
    Load Hugging Face token by account role (`professional` vs `showcase`):
    - Inside an HF Space container, `HF_TOKEN` / `HUGGINGFACE_API_TOKEN` env vars take precedence.
    - Locally, resolves either `DevAnupam2008` (`professional`) or `AnupamSahoo` (`showcase`) from the central vault.
    """
    mode = str(account or "professional").strip().lower()
    if mode in ("showcase", "anupamsahoo"):
        env_tok = os.environ.get("HUGGINGFACE_SHOWCASE_API_TOKEN") or os.environ.get("HUGGINGFACE_SHOWCASE_TOKEN")
        if env_tok:
            return env_tok.strip()
        return _read_vault_key(("HUGGINGFACE_SHOWCASE_API_TOKEN", "HUGGINGFACE_SHOWCASE_TOKEN"))

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HUGGINGFACE_PROFESSIONAL_TOKEN")
    if token:
        return token.strip()
    return _read_vault_key(("HUGGINGFACE_PROFESSIONAL_TOKEN", "HUGGINGFACE_API_TOKEN", "HF_TOKEN"))


def _load_github_token() -> str | None:
    """Load GitHub token from environment or central vault."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    return _read_vault_key(("GITHUB_TOKEN",))


HF_PROFESSIONAL_USERNAME = os.environ.get("HUGGINGFACE_PROFESSIONAL_USERNAME", "DevAnupam2008")
HF_SHOWCASE_USERNAME = os.environ.get("HUGGINGFACE_SHOWCASE_USERNAME", "AnupamSahoo")
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "anupamsahoo4508")

HF_TOKEN = _load_hf_token("professional")
HF_SHOWCASE_TOKEN = _load_hf_token("showcase")
GITHUB_TOKEN = _load_github_token()

# Detect Public Developer Showcase Space (`AnupamSahoo/...` or `PASSPORT_SHOWCASE_MODE=1`)
# where no private customer Dataset is used (preview-only mode on public ZeroGPU).
_ACTIVE_SPACE_ID = os.environ.get("SPACE_ID", "").strip()
IS_SHOWCASE_MODE: bool = bool(
    os.environ.get("PASSPORT_SHOWCASE_MODE", "0").strip().lower() in ("1", "true", "yes")
    or _ACTIVE_SPACE_ID.lower().startswith("anupamsahoo/")
)

REPO_ID = "" if IS_SHOWCASE_MODE else os.environ.get("STUDIO_DB_REPO_ID", "DevAnupam2008/Studio-Database")
SPACE_REPO_ID = os.environ.get(
    "STUDIO_SPACE_REPO_ID",
    _ACTIVE_SPACE_ID if _ACTIVE_SPACE_ID else "DevAnupam2008/Passport-Studio-V2",
)
SHOWCASE_SPACE_REPO_ID = os.environ.get("SHOWCASE_SPACE_REPO_ID", "AnupamSahoo/Advance-Passport-Photo-Studio")
PUBLIC_PORTAL_SPACE_ID = "" if IS_SHOWCASE_MODE else os.environ.get("PUBLIC_PORTAL_SPACE_ID", "DevAnupam2008/Maa-Sitala-Portal")
PUBLIC_PORTAL_BASE_URL = (
    ""
    if IS_SHOWCASE_MODE
    else os.environ.get(
        "PUBLIC_PORTAL_BASE_URL",
        "https://devanupam2008-maa-sitala-portal.static.hf.space",
    ).rstrip("/")
)

# === Cloud-Only Model Execution Guard ===
# Strictly TRUE only inside Hugging Face Cloud containers (`SPACE_ID` / `SPACE_HOST`), NEVER on local Windows PC!
IS_HF_CLOUD: bool = bool(
    os.environ.get("SPACE_ID")
    or os.environ.get("SPACE_HOST")
    or (os.name != "nt" and os.environ.get("PASSPORT_CLOUD_RUNTIME", "0").strip() == "1")
)
CLOUD_MODELS_DIR = os.environ.get("PASSPORT_CLOUD_MODELS_DIR", "/tmp/passport_studio_models")


def get_portal_hmac_secret() -> bytes:
    """Derive a cryptographic 256-bit HMAC key shared between Private Studio and Public Gateway."""
    explicit = os.environ.get("PORTAL_HMAC_SECRET", "").strip()
    seed = explicit or (HF_TOKEN or "maa-sitala-local-dev-only")
    return hashlib.pbkdf2_hmac(
        "sha256",
        seed.encode("utf-8"),
        b"maa-sitala-zero-trust-portal-v7",
        60_000,
        dklen=32,
    )

# === Directory Structure ===
BASE_DATA_DIR = os.environ.get("PASSPORT_STUDIO_DATA_DIR", "shop_data")
DB_DIR = os.path.join(BASE_DATA_DIR, "database")
TEMP_DIR = os.path.join(BASE_DATA_DIR, "temp")
BACKUP_DIR = os.path.join(BASE_DATA_DIR, "backups")
HISTORY_DIR = os.path.join(BASE_DATA_DIR, "history")
IMG_DIR = os.path.join(DB_DIR, "customer_images")
THUMB_DIR = os.path.join(DB_DIR, "customer_thumbs")
DB_PATH = os.path.join(DB_DIR, "shop_customers.db")

# === Limits ===
MAX_STORAGE_MB = 3000.0
MAX_UPLOAD_SIZE_MB = 15

# === ICAO 9303 & Studio Resolution Standards ===
PASSPORT_STANDARDS = {
    "india_passport_35x45mm": {
        "label": "🇮🇳 Indian Passport / Visa (35×45 mm)",
        "aspect_w": 35,
        "aspect_h": 45,
        "px_300dpi": (413, 531),
        "px_600dpi": (827, 1063),
        "px_4k": (2987, 3840),
        "head_ratio_target": 0.62,   # ICAO 9303: 50% - 69% crown-to-chin
        "eye_line_from_bottom": 0.61 # ICAO 9303: 56% - 69% from bottom edge
    },
    "us_visa_2x2in": {
        "label": "🇺🇸 US Visa / OCI (2×2 inch)",
        "aspect_w": 51,
        "aspect_h": 51,
        "px_300dpi": (600, 600),
        "px_600dpi": (1200, 1200),
        "px_4k": (3840, 3840),
        "head_ratio_target": 0.59,
        "eye_line_from_bottom": 0.62
    },
    "uk_eu_schengen_35x45mm": {
        "label": "🇪🇺 UK / EU Passport (35×45 mm)",
        "aspect_w": 35,
        "aspect_h": 45,
        "px_300dpi": (413, 531),
        "px_600dpi": (827, 1063),
        "px_4k": (2987, 3840),
        "head_ratio_target": 0.64,
        "eye_line_from_bottom": 0.62
    },
    "india_pan_stamp_25x35mm": {
        "label": "🪪 PAN Card / Stamp Size (25×35 mm)",
        "aspect_w": 25,
        "aspect_h": 35,
        "px_300dpi": (295, 413),
        "px_600dpi": (591, 827),
        "px_4k": (2743, 3840),
        "head_ratio_target": 0.64,
        "eye_line_from_bottom": 0.60
    },
    "original_ratio": {
        "label": "📐 Keep Original Size (No Crop)",
        "aspect_w": None,
        "aspect_h": None,
        "px_300dpi": None,
        "px_600dpi": None,
        "px_4k": (3840, 3840),
        "head_ratio_target": 0.62,
        "eye_line_from_bottom": 0.61
    }
}

# === Create all directories on import ===
for _dir in [DB_DIR, TEMP_DIR, BACKUP_DIR, HISTORY_DIR, IMG_DIR, THUMB_DIR]:
    os.makedirs(_dir, exist_ok=True)
