"""
Mobile QR Camera Shoot (`?mobile_shoot=<session_id>`) & Permanent 4-Digit PIN Customer Share Vault (`?share=<share_id>`).
Includes:
1. Self-Contained Pure-Python / NumPy ISO/IEC 18004 QR Code Encoder & Verifier (GF(256) Reed-Solomon EC,
   Finder/Alignment/Timing patterns, BCH(15,5) format polynomial, and SVG/RGB rendering — zero external deps).
2. Direct Phone Camera Shoot Session Manager (`create_mobile_shoot_session`, `save_mobile_shoot_photo`,
   `poll_mobile_shoot_photo`, `poll_mobile_shoot_photo_status`).
3. Permanent 4-Digit PIN Customer Share Vault (`create_customer_share`, `verify_customer_share_pin`,
   `get_customer_share_meta`, `render_interactive_deep_zoom_html`) backed by SQLite + `DevAnupam2008/Studio-Database`.
"""
from __future__ import annotations
import os
import re
import io
import json
import time
import hmac
import base64
import socket
import sqlite3
import secrets
import hashlib
import threading
import numpy as np
import urllib.request
from PIL import Image, ImageOps
from src.shared.config import (
    DB_DIR,
    DB_PATH,
    TEMP_DIR,
    HF_TOKEN,
    REPO_ID,
    PUBLIC_PORTAL_SPACE_ID,
    PUBLIC_PORTAL_BASE_URL,
    get_portal_hmac_secret,
)
from src.shared.cloud_database import init_db, push_to_cloud
from src.core.cv_ops import imread_rgb, imwrite_rgb

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_AESGCM = True
except ImportError:
    AESGCM = None
    HAS_AESGCM = False

try:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # type: ignore
    HAS_HF_HUB = True
except ImportError:
    CommitOperationAdd = None
    HfApi = None
    hf_hub_download = None
    HAS_HF_HUB = False

MOBILE_SHOOT_DIR = os.path.join(TEMP_DIR, "mobile_sessions")
SHARES_DIR = os.path.join(DB_DIR, "shares")
SHARES_INDEX_PATH = os.path.join(SHARES_DIR, "shares_index.json")

_MOBILE_SHOOT_STORE: dict[str, dict] = {}
_CONSUMED_MOBILE_MTIMES: dict[str, float] = {}
_LAST_PORTAL_POLL_TS: dict[str, float] = {}
_REMOTE_SHOOT_UPDATED: dict[str, float] = {}
_REMOTE_SHOOT_KEYS: dict[str, str] = {}
_CHUNK_BYTES_CACHE: dict[str, bytes] = {}
_LATEST_MOBILE_SESSION_ID: str | None = None
_STORE_LOCK = threading.Lock()


def _portal_sign(message: str, hex_len: int = 32) -> str:
    """Compute HMAC-SHA256 hex prefix matching `portal_space/portal_app.py`."""
    return hmac.new(
        get_portal_hmac_secret(),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:hex_len]


def _is_offline_test_mode() -> bool:
    return os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip().lower() in ("1", "true", "yes")


# ============================================================================
# 1. PURE-PYTHON / NUMPY ISO/IEC 18004 QR CODE GENERATOR & VERIFIER (VERSIONS 1..10, EC LEVEL L)
# ============================================================================

# ISO/IEC 18004 Table 9 — Level L (7% Error Correction) parameters:
# version: (size, total_data_codewords, ec_codewords_per_block, num_blocks, alignment_centers)
_QR_VERSION_SPEC: dict[int, tuple[int, int, int, int, tuple[int, ...]]] = {
    1:  (21,  19,  7, 1, ()),
    2:  (25,  34, 10, 1, (6, 18)),
    3:  (29,  55, 15, 1, (6, 22)),
    4:  (33,  80, 20, 1, (6, 26)),
    5:  (37, 108, 26, 1, (6, 30)),
    6:  (41, 136, 18, 2, (6, 34)),
    7:  (45, 156, 20, 2, (6, 22, 38)),
    8:  (49, 194, 24, 2, (6, 24, 42)),
    9:  (53, 232, 30, 2, (6, 26, 46)),
    10: (57, 274, 18, 4, (6, 28, 50)),
}

# Precompute GF(256) log/exp tables with primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1)
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _GF_EXP[_i] = _GF_EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator_poly(nsym: int) -> list[int]:
    g = [1]
    for i in range(nsym):
        ng = [0] * (len(g) + 1)
        root = _GF_EXP[i]
        for j, coef in enumerate(g):
            ng[j] ^= coef
            ng[j + 1] ^= _gf_mul(coef, root)
        g = ng
    return g


def _rs_encode(data_bytes: list[int], nsym: int) -> list[int]:
    gen = _rs_generator_poly(nsym)
    res = list(data_bytes) + [0] * nsym
    for i in range(len(data_bytes)):
        coef = res[i]
        if coef != 0:
            for j in range(1, len(gen)):
                res[i + j] ^= _gf_mul(gen[j], coef)
    return res[len(data_bytes):]


def _rs_verify_syndromes(codewords: list[int], nsym: int) -> bool:
    """Verify that all Reed-Solomon syndromes S_i = C(alpha^i) == 0 for i = 0..nsym-1."""
    for i in range(nsym):
        root = _GF_EXP[i]
        syn = 0
        for cw in codewords:
            syn = _gf_mul(syn, root) ^ cw
        if syn != 0:
            return False
    return True


def _compute_bch_format_bits(ec_indicator: int, mask_pattern: int) -> int:
    """Compute 15-bit ISO/IEC 18004 Format Information with BCH(15, 5) and XOR mask 0x5412."""
    data = ((ec_indicator & 0x3) << 3) | (mask_pattern & 0x7)
    rem = data << 10
    for i in range(4, -1, -1):
        if rem & (1 << (i + 10)):
            rem ^= 0x537 << i
    return ((data << 10) | rem) ^ 0x5412


def _compute_golay_version_bits(version: int) -> int:
    """Compute 18-bit ISO/IEC 18004 Version Information with Golay(18, 6) polynomial 0x1F25."""
    rem = version << 12
    for i in range(5, -1, -1):
        if rem & (1 << (i + 12)):
            rem ^= 0x1F25 << i
    return (version << 12) | rem


def _build_reserved_mask(size: int, chosen_ver: int, align_centers: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Construct the function-pattern grid and reserved boolean mask for a given QR version."""
    grid = np.zeros((size, size), dtype=np.uint8)
    reserved = np.zeros((size, size), dtype=bool)

    def _place_finder(top: int, left: int) -> None:
        for r in range(-1, 8):
            for c in range(-1, 8):
                rr, cc = top + r, left + c
                if 0 <= rr < size and 0 <= cc < size:
                    reserved[rr, cc] = True
                    if 0 <= r <= 6 and 0 <= c <= 6:
                        if r in (0, 6) or c in (0, 6) or (2 <= r <= 4 and 2 <= c <= 4):
                            grid[rr, cc] = 1
                        else:
                            grid[rr, cc] = 0
                    else:
                        grid[rr, cc] = 0

    _place_finder(0, 0)
    _place_finder(0, size - 7)
    _place_finder(size - 7, 0)

    # Alignment patterns
    for ar in align_centers:
        for ac in align_centers:
            if reserved[ar, ac]:
                continue
            for r in range(-2, 3):
                for c in range(-2, 3):
                    rr, cc = ar + r, ac + c
                    reserved[rr, cc] = True
                    if max(abs(r), abs(c)) in (2, 0):
                        grid[rr, cc] = 1
                    else:
                        grid[rr, cc] = 0

    # Timing patterns (row 6 and col 6)
    for i in range(8, size - 8):
        if not reserved[6, i]:
            reserved[6, i] = True
            grid[6, i] = 1 if (i % 2 == 0) else 0
        if not reserved[i, 6]:
            reserved[i, 6] = True
            grid[i, 6] = 1 if (i % 2 == 0) else 0

    # Dark module & format info reservation
    reserved[4 * chosen_ver + 9, 8] = True
    grid[4 * chosen_ver + 9, 8] = 1

    for i in range(9):
        if i < size:
            reserved[8, i] = True
            reserved[i, 8] = True
    for i in range(8):
        reserved[8, size - 1 - i] = True
        reserved[size - 1 - i, 8] = True

    if chosen_ver >= 7:
        vbits = _compute_golay_version_bits(chosen_ver)
        bit_k = 0
        for i in range(6):
            for j in range(3):
                b = (vbits >> bit_k) & 1
                bit_k += 1
                reserved[i, size - 11 + j] = True
                grid[i, size - 11 + j] = b
                reserved[size - 11 + j, i] = True
                grid[size - 11 + j, i] = b

    return grid, reserved


def encode_qr_matrix(text: str, border: int = 4) -> np.ndarray:
    """
    Encode arbitrary UTF-8 string into an ISO/IEC 18004 QR Code binary matrix (uint8, 1 = dark, 0 = light)
    with a quiet zone of `border` modules around all 4 sides.
    """
    payload = text.encode("utf-8")
    n_bytes = len(payload)

    # Select smallest QR version (1..10) that fits payload in Byte Mode
    # Note: Versions 1..9 use 8-bit character count indicator; Version 10 uses 16-bit character count indicator
    chosen_ver = 10
    for v, (_sz, total_dcw, _ec, _nb, _align) in _QR_VERSION_SPEC.items():
        hdr_bytes = 2 if v <= 9 else 3
        if n_bytes + hdr_bytes <= total_dcw:
            chosen_ver = v
            break
    max_cap = _QR_VERSION_SPEC[10][1] - 3
    if n_bytes > max_cap:
        payload = payload[:max_cap]
        n_bytes = len(payload)

    size, total_dcw, ec_per_block, num_blocks, align_centers = _QR_VERSION_SPEC[chosen_ver]

    # Build bit stream: Byte Mode indicator (0100) + character count (8-bit for v<=9, 16-bit for v>=10) + byte bits + terminator
    bits: list[int] = [0, 1, 0, 0]
    count_bits = 8 if chosen_ver <= 9 else 16
    for shift in range(count_bits - 1, -1, -1):
        bits.append((n_bytes >> shift) & 1)
    for b in payload:
        for shift in range(7, -1, -1):
            bits.append((b >> shift) & 1)

    max_bits = total_dcw * 8
    term_len = min(4, max_bits - len(bits))
    bits.extend([0] * term_len)
    while len(bits) % 8 != 0:
        bits.append(0)

    data_codewords: list[int] = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for bit in bits[i : i + 8]:
            byte_val = (byte_val << 1) | bit
        data_codewords.append(byte_val)

    pad_bytes = (0xEC, 0x11)
    pad_idx = 0
    while len(data_codewords) < total_dcw:
        data_codewords.append(pad_bytes[pad_idx % 2])
        pad_idx += 1

    # Reed-Solomon encoding & block interleaving (handles equal and unequal block sizes e.g. Version 10-L)
    if num_blocks == 1:
        ec_codewords = _rs_encode(data_codewords, ec_per_block)
        final_codewords = data_codewords + ec_codewords
    else:
        short_len = total_dcw // num_blocks
        num_long = total_dcw % num_blocks
        num_short = num_blocks - num_long
        data_blocks: list[list[int]] = []
        offset = 0
        for b_idx in range(num_blocks):
            blen = short_len + (1 if b_idx >= num_short else 0)
            data_blocks.append(data_codewords[offset : offset + blen])
            offset += blen
        ec_blocks = [_rs_encode(db, ec_per_block) for db in data_blocks]
        final_codewords = []
        max_blk_len = max(len(db) for db in data_blocks)
        for col in range(max_blk_len):
            for b_idx in range(num_blocks):
                if col < len(data_blocks[b_idx]):
                    final_codewords.append(data_blocks[b_idx][col])
        for col in range(ec_per_block):
            for b_idx in range(num_blocks):
                final_codewords.append(ec_blocks[b_idx][col])

    all_bits: list[int] = []
    for cw in final_codewords:
        for shift in range(7, -1, -1):
            all_bits.append((cw >> shift) & 1)
    # Remainder bits (7 bits for versions 2..6)
    if 2 <= chosen_ver <= 6:
        all_bits.extend([0] * 7)

    grid, reserved = _build_reserved_mask(size, chosen_ver, align_centers)

    # Zig-zag upward/downward 2-column data placement + Mask Pattern 0: (r + c) % 2 == 0
    bit_idx = 0
    upward = True
    col = size - 1
    while col > 0:
        if col == 6:
            col -= 1
        row_iter = range(size - 1, -1, -1) if upward else range(size)
        for r in row_iter:
            for c_off in (0, 1):
                c = col - c_off
                if not reserved[r, c]:
                    b = all_bits[bit_idx] if bit_idx < len(all_bits) else 0
                    bit_idx += 1
                    if (r + c) % 2 == 0:
                        b ^= 1
                    grid[r, c] = b
        upward = not upward
        col -= 2

    # Place 15-bit Format Information (Error Correction Level L = 0b01, Mask = 0b000)
    # Per ISO/IEC 18004 Figure 25: bit 14 (MSB) is placed at (8, 0) and (size - 1, 8),
    # descending down to bit 0 (LSB) at (0, 8) and (8, size - 1).
    fmt = _compute_bch_format_bits(ec_indicator=0b01, mask_pattern=0)
    fmt_coords_1 = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    fmt_coords_2 = [
        (size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
        (size - 5, 8), (size - 6, 8), (size - 7, 8),
        (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5),
        (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1),
    ]
    for i in range(15):
        b = (fmt >> (14 - i)) & 1
        r1, c1 = fmt_coords_1[i]
        r2, c2 = fmt_coords_2[i]
        grid[r1, c1] = b
        grid[r2, c2] = b

    if border > 0:
        padded = np.zeros((size + 2 * border, size + 2 * border), dtype=np.uint8)
        padded[border : border + size, border : border + size] = grid
        return padded
    return grid


def decode_qr_matrix(mat: np.ndarray) -> str:
    """
    Pure-Python ISO/IEC 18004 QR Code Decoder & Mathematical Syndrome Verifier.
    Strips quiet-zone borders, verifies both copies of the 15-bit BCH(15, 5) Format Information,
    unmasks the data modules, de-interleaves the Reed-Solomon blocks, verifies that all RS error
    syndromes equal zero, and decodes the UTF-8 Byte-mode payload.
    """
    # Strip quiet zone if present
    rows, cols = np.where(mat > 0)
    if len(rows) == 0:
        raise ValueError("Empty QR matrix")
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    core = mat[r0 : r1 + 1, c0 : c1 + 1]
    size = core.shape[0]
    if core.shape[0] != core.shape[1] or (size - 21) % 4 != 0:
        raise ValueError(f"Invalid QR core size: {core.shape}")
    version = (size - 17) // 4
    if version not in _QR_VERSION_SPEC:
        raise ValueError(f"Unsupported QR version: {version}")

    _, total_dcw, ec_per_block, num_blocks, align_centers = _QR_VERSION_SPEC[version]

    # Read 15-bit Format Information from both copies (bit 14 MSB .. bit 0 LSB)
    fmt_coords_1 = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    fmt_coords_2 = [
        (size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
        (size - 5, 8), (size - 6, 8), (size - 7, 8),
        (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5),
        (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1),
    ]
    fmt1 = 0
    fmt2 = 0
    for r, c in fmt_coords_1:
        fmt1 = (fmt1 << 1) | int(core[r, c])
    for r, c in fmt_coords_2:
        fmt2 = (fmt2 << 1) | int(core[r, c])
    if fmt1 != fmt2:
        raise ValueError(f"Format info copies mismatch: {fmt1:#06x} != {fmt2:#06x}")

    unmasked_fmt = fmt1 ^ 0x5412
    ec_ind = (unmasked_fmt >> 13) & 0x3
    mask_pat = (unmasked_fmt >> 10) & 0x7
    if fmt1 != _compute_bch_format_bits(ec_ind, mask_pat):
        raise ValueError(f"Invalid BCH(15, 5) format word: {fmt1:#06x}")

    _, reserved = _build_reserved_mask(size, version, align_centers)
    raw_bits: list[int] = []
    upward = True
    col = size - 1
    while col > 0:
        if col == 6:
            col -= 1
        row_iter = range(size - 1, -1, -1) if upward else range(size)
        for r in row_iter:
            for c_off in (0, 1):
                c = col - c_off
                if not reserved[r, c]:
                    b = int(core[r, c])
                    if mask_pat == 0 and (r + c) % 2 == 0:
                        b ^= 1
                    raw_bits.append(b)
        upward = not upward
        col -= 2

    total_cw = total_dcw + num_blocks * ec_per_block
    codewords: list[int] = []
    for i in range(total_cw):
        val = 0
        for bit in raw_bits[i * 8 : (i + 1) * 8]:
            val = (val << 1) | bit
        codewords.append(val)

    # De-interleave blocks and verify Reed-Solomon syndromes
    if num_blocks == 1:
        if not _rs_verify_syndromes(codewords, ec_per_block):
            raise ValueError("Reed-Solomon syndrome check failed on block 0")
        data_cw = codewords[:total_dcw]
    else:
        short_len = total_dcw // num_blocks
        num_long = total_dcw % num_blocks
        num_short = num_blocks - num_long
        data_blocks = [[] for _ in range(num_blocks)]
        ec_blocks = [[] for _ in range(num_blocks)]
        idx = 0
        max_blk_len = short_len + (1 if num_long > 0 else 0)
        for col_i in range(max_blk_len):
            for b_idx in range(num_blocks):
                blen = short_len + (1 if b_idx >= num_short else 0)
                if col_i < blen:
                    data_blocks[b_idx].append(codewords[idx])
                    idx += 1
        for col_i in range(ec_per_block):
            for b_idx in range(num_blocks):
                ec_blocks[b_idx].append(codewords[idx])
                idx += 1
        data_cw = []
        for b_idx in range(num_blocks):
            full_blk = data_blocks[b_idx] + ec_blocks[b_idx]
            if not _rs_verify_syndromes(full_blk, ec_per_block):
                raise ValueError(f"Reed-Solomon syndrome check failed on block {b_idx}")
            data_cw.extend(data_blocks[b_idx])

    data_bits: list[int] = []
    for cw in data_cw:
        for shift in range(7, -1, -1):
            data_bits.append((cw >> shift) & 1)

    mode = (data_bits[0] << 3) | (data_bits[1] << 2) | (data_bits[2] << 1) | data_bits[3]
    if mode != 0b0100:
        raise ValueError(f"Expected Byte Mode (0100), got {mode:04b}")
    count_bits = 8 if version <= 9 else 16
    n_bytes = 0
    for bit in data_bits[4 : 4 + count_bits]:
        n_bytes = (n_bytes << 1) | bit
    pos = 4 + count_bits
    out_bytes = bytearray()
    for _ in range(n_bytes):
        val = 0
        for bit in data_bits[pos : pos + 8]:
            val = (val << 1) | bit
        out_bytes.append(val)
        pos += 8
    return out_bytes.decode("utf-8")


def generate_qr_image_rgb(
    url: str,
    size: int = 280,
    box_size: int | None = None,
    border: int = 4,
) -> np.ndarray:
    """
    Render an ISO/IEC 18004 QR Code as a crisp RGB uint8 numpy image (white background, deep navy-black modules).
    Supports both `size` (target pixel dimension, default 280) and `box_size` (module pixel multiplier).
    """
    mat = encode_qr_matrix(url, border=border)
    h_m, w_m = mat.shape
    if box_size is not None:
        scale = max(2, int(box_size))
        upscaled = np.repeat(np.repeat(mat, scale, axis=0), scale, axis=1)
    elif size <= 24:
        scale = max(2, int(size))
        upscaled = np.repeat(np.repeat(mat, scale, axis=0), scale, axis=1)
    else:
        target_px = max(h_m * 2, int(size))
        scale = max(2, target_px // h_m)
        upscaled = np.repeat(np.repeat(mat, scale, axis=0), scale, axis=1)
        if upscaled.shape[0] != target_px:
            idx = np.clip((np.arange(target_px) * h_m) // target_px, 0, h_m - 1)
            upscaled = mat[idx][:, idx]
    rgb = np.where(
        upscaled[:, :, None] == 1,
        np.array([15, 23, 42], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    return rgb


def generate_qr_svg_data_uri(url: str, border: int = 3) -> str:
    """
    Render an ISO/IEC 18004 QR Code as a compact `data:image/svg+xml;base64,...` URI for instant HTML/JS popups.
    """
    mat = encode_qr_matrix(url, border=border)
    n = mat.shape[0]
    rects = []
    for r in range(n):
        for c in range(n):
            if mat[r, c] == 1:
                rects.append(f"<rect x='{c}' y='{r}' width='1' height='1'/>")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {n} {n}' shape-rendering='crispEdges'>"
        f"<rect width='{n}' height='{n}' fill='#ffffff'/>"
        f"<g fill='#0f172a'>{''.join(rects)}</g></svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


# ============================================================================
# 2. PUBLIC / LAN URL DISCOVERY FOR PHONE QR SCANNING
# ============================================================================

def _detect_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def get_public_studio_base_url(request=None) -> str:
    """
    Resolve the externally reachable base URL of the studio so a phone scanning the QR code
    opens the live Hugging Face Space, Cloudflare Tunnel, or LAN server directly.
    """
    env_url = (
        os.environ.get("PASSPORT_PUBLIC_BASE_URL")
        or os.environ.get("GRADIO_PUBLIC_URL")
        or ""
    ).strip()
    if env_url:
        return env_url.rstrip("/")

    space_host = os.environ.get("SPACE_HOST", "").strip()
    if space_host:
        if space_host.startswith("http"):
            return space_host.rstrip("/")
        return f"https://{space_host.rstrip('/')}"

    if request is not None:
        try:
            headers = getattr(request, "headers", {}) or {}
            host = (
                headers.get("x-forwarded-host")
                or headers.get("host")
                or ""
            ).strip()
            proto = (headers.get("x-forwarded-proto") or "https").strip()
            if host and not host.startswith("127.0.0.1") and not host.startswith("localhost"):
                return f"{proto}://{host.rstrip('/')}"
            if host:
                lan_ip = _detect_lan_ip()
                port = host.split(":")[-1] if ":" in host else "7860"
                return f"http://{lan_ip}:{port}"
        except Exception:
            pass

    lan_ip = _detect_lan_ip()
    port = os.environ.get("GRADIO_SERVER_PORT", "7860")
    return f"http://{lan_ip}:{port}"


# ============================================================================
# 3. DIRECT PHONE QR CAMERA SHOOT ENGINE (?mobile_shoot=<session_id>)
# ============================================================================

def _sanitize_token(val: str, prefix: str = "ms") -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", str(val or "").strip())
    return clean[:64] if clean else f"{prefix}_{secrets.token_hex(4)}"


def create_mobile_shoot_session(base_url: str | None = None, request=None) -> dict:
    """
    Create a new Zero-HF-Login End-to-End AES-256-GCM Phone Camera QR Shoot session
    routed via `PUBLIC_PORTAL_BASE_URL/shoot.html#s={session_id}&k={k_hex}`
    unless an explicit `base_url` is supplied.
    Returns session_id, shoot_url, slot_path (`shop_data/temp/mobile_sessions/{session_id}.png`),
    qr_rgb (uint8 HxWx3), and qr_svg_uri.
    """
    global _LATEST_MOBILE_SESSION_ID
    os.makedirs(MOBILE_SHOOT_DIR, exist_ok=True)
    session_id = f"ms_{secrets.token_hex(5)}"
    k_hex = _portal_sign(f"shoot_key:{session_id}", hex_len=32)

    if base_url and str(base_url).strip():
        resolved_base = str(base_url).strip().rstrip("/")
        shoot_url = f"{resolved_base}/?mobile_shoot={session_id}"
    elif PUBLIC_PORTAL_BASE_URL:
        portal_base = PUBLIC_PORTAL_BASE_URL.rstrip("/")
        shoot_url = f"{portal_base}/shoot.html#s={session_id}&k={k_hex}"
    else:
        resolved_base = get_public_studio_base_url(request)
        shoot_url = f"{resolved_base}/?mobile_shoot={session_id}"

    qr_rgb = generate_qr_image_rgb(shoot_url, size=280, border=4)
    qr_svg_uri = generate_qr_svg_data_uri(shoot_url, border=3)
    slot_path = os.path.abspath(os.path.join(MOBILE_SHOOT_DIR, f"{session_id}.png"))

    with _STORE_LOCK:
        _LATEST_MOBILE_SESSION_ID = session_id
        _MOBILE_SHOOT_STORE[session_id] = {
            "session_id": session_id,
            "shoot_url": shoot_url,
            "slot_path": slot_path,
            "k_hex": k_hex,
            "created_at": time.time(),
            "photo_path": None,
            "updated_at": None,
        }
    return {
        "session_id": session_id,
        "shoot_url": shoot_url,
        "slot_path": slot_path,
        "k_hex": k_hex,
        "qr_rgb": qr_rgb,
        "qr_svg_uri": qr_svg_uri,
    }


def save_mobile_shoot_photo(session_id: str, img_rgb: np.ndarray) -> str:
    """
    Save a photo shot on the user's phone camera (`?mobile_shoot=<session_id>`) into
    `shop_data/temp/mobile_sessions/{session_id}.png` so the desktop studio can immediately
    load it into `Upload Photo`.
    """
    global _LATEST_MOBILE_SESSION_ID
    if img_rgb is None or not isinstance(img_rgb, np.ndarray) or img_rgb.ndim < 2:
        raise ValueError("Invalid mobile camera image")
    os.makedirs(MOBILE_SHOOT_DIR, exist_ok=True)
    clean_sid = _sanitize_token(session_id, prefix="shoot")
    rgb = img_rgb[:, :, :3].copy() if img_rgb.ndim == 3 else np.stack([img_rgb] * 3, axis=-1)
    file_path = os.path.abspath(os.path.join(MOBILE_SHOOT_DIR, f"{clean_sid}.png"))
    imwrite_rgb(file_path, rgb)

    with _STORE_LOCK:
        _LATEST_MOBILE_SESSION_ID = clean_sid
        _MOBILE_SHOOT_STORE[clean_sid] = {
            "session_id": clean_sid,
            "slot_path": file_path,
            "photo_path": file_path,
            "updated_at": time.time(),
        }
    return file_path


# Maximum per-PUT attachment size (850 KB — safely <50% of ntfy.sh's 2 MB HTTP 413 ceiling!)
MSV8_MAX_CHUNK_BYTES = 870_400


def slice_envelope_msv8(
    msv7_envelope: bytes,
    shot_id_bytes: bytes | None = None,
    max_chunk_bytes: int = MSV8_MAX_CHUNK_BYTES,
) -> list[tuple[str, bytes]]:
    """
    Zero-413 Encrypted Envelope Slicer (`MSV7` single-packet fast path + `MSV8` multi-chunk stream):
    - If `len(msv7_envelope) <= max_chunk_bytes`, emits a single `(filename, msv7_envelope)` packet.
    - Otherwise splits `msv7_envelope` into `<= max_chunk_bytes` slices, each prefixed with a 20-byte
      `MSV8` framing header: `b"MSV8" | shot_id(8B) | chunk_idx_u16be(2B) | total_u16be(2B) | total_len_u32be(4B)`.
    """
    sid8 = shot_id_bytes[:8] if (shot_id_bytes and len(shot_id_bytes) >= 8) else secrets.token_bytes(8)
    sid_hex = sid8.hex()
    total_len = len(msv7_envelope)
    payload_cap = max(1024, int(max_chunk_bytes) - 20)
    if total_len <= max_chunk_bytes:
        return [(f"msv7_{sid_hex}_1of1.bin", msv7_envelope)]

    n_chunks = (total_len + payload_cap - 1) // payload_cap
    packets: list[tuple[str, bytes]] = []
    for idx in range(n_chunks):
        part = msv7_envelope[idx * payload_cap : (idx + 1) * payload_cap]
        hdr = (
            b"MSV8"
            + sid8
            + idx.to_bytes(2, "big")
            + n_chunks.to_bytes(2, "big")
            + total_len.to_bytes(4, "big")
        )
        packets.append((f"msv8_{sid_hex}_{idx + 1}of{n_chunks}.bin", hdr + part))
    return packets


def reassemble_msv8_or_msv7_packets(raw_packets: list[bytes], aes_key: bytes) -> bytes:
    """
    Reassemble `1..N` `MSV8` encrypted chunks (or a single `MSV7` packet) in exact index order,
    verify total byte length & `MSV7` magic header, and authenticate/decrypt `AES-256-GCM` payload.
    """
    if not raw_packets:
        raise ValueError("Empty mobile packet list")
    if not HAS_AESGCM or AESGCM is None:
        raise RuntimeError("AESGCM cryptography backend unavailable")

    if len(raw_packets) == 1 and len(raw_packets[0]) >= 32 and raw_packets[0][:4] == b"MSV7":
        full_env = raw_packets[0]
    else:
        parsed_parts: dict[int, bytes] = {}
        expected_total = 0
        expected_len = 0
        ref_sid = None
        for pkt in raw_packets:
            if len(pkt) < 21 or pkt[:4] != b"MSV8":
                raise ValueError("Invalid MSV8 chunk header")
            sid8 = pkt[4:12]
            c_idx = int.from_bytes(pkt[12:14], "big")
            c_tot = int.from_bytes(pkt[14:16], "big")
            t_len = int.from_bytes(pkt[16:20], "big")
            if ref_sid is None:
                ref_sid = sid8
                expected_total = c_tot
                expected_len = t_len
            elif sid8 != ref_sid or c_tot != expected_total or t_len != expected_len:
                raise ValueError("Mismatched MSV8 shot metadata across chunks")
            parsed_parts[c_idx] = pkt[20:]

        if expected_total < 1 or len(parsed_parts) != expected_total:
            raise ValueError(f"Incomplete MSV8 stream ({len(parsed_parts)}/{expected_total} chunks)")
        full_env = b"".join(parsed_parts[i] for i in range(expected_total))
        if len(full_env) != expected_len:
            raise ValueError(f"MSV8 length mismatch: got {len(full_env)}, expected {expected_len}")

    if len(full_env) < 32 or full_env[:4] != b"MSV7":
        raise ValueError("Reassembled payload missing MSV7 AES-GCM envelope header")
    iv12 = full_env[4:16]
    ct_bytes = full_env[16:]
    return AESGCM(aes_key).decrypt(iv12, ct_bytes, None)


_MSV_FNAME_RE = re.compile(r"^(msv[78])_([0-9a-fA-F]{8,32})_(\d+)of(\d+)\.bin$")


def _download_relay_attachment_cached(att_url: str) -> bytes:
    with _STORE_LOCK:
        cached = _CHUNK_BYTES_CACHE.get(att_url)
    if cached is not None:
        return cached
    att_req = urllib.request.Request(att_url, headers={"User-Agent": "MaaSitalaStudioV8/1.0"})
    with urllib.request.urlopen(att_req, timeout=20.0) as att_resp:
        raw = att_resp.read()
    with _STORE_LOCK:
        if len(_CHUNK_BYTES_CACHE) > 48:
            _CHUNK_BYTES_CACHE.clear()
        _CHUNK_BYTES_CACHE[att_url] = raw
    return raw


def _try_pull_shoot_from_portal(target_sid: str, force: bool = False) -> str | None:
    """
    Pull and decrypt newly snapped `AES-256-GCM` mobile camera portrait from the E2EE relay
    associated with `target_sid` (supports both `MSV7` single-packet fast path and `MSV8`
    multi-chunk large HD uploads with ZERO `HTTP 413` failures, plus multi-customer continuous shots!).
    """
    if not target_sid or _is_offline_test_mode() or not HAS_AESGCM:
        return None

    now = time.time()
    with _STORE_LOCK:
        last_poll = _LAST_PORTAL_POLL_TS.get(target_sid, 0.0)
        if not force and (now - last_poll) < 1.35:
            return None
        _LAST_PORTAL_POLL_TS[target_sid] = now

    k_hex = _portal_sign(f"shoot_key:{target_sid}", hex_len=32)
    topic = "ms_cam_" + hashlib.sha256(f"topic:{target_sid}:{k_hex}".encode("utf-8")).hexdigest()[:26]
    aes_key = hashlib.sha256(f"aes256:{target_sid}:{k_hex}".encode("utf-8")).digest()
    poll_url = f"https://ntfy.sh/{topic}/json?poll=1&since=2h"

    try:
        req = urllib.request.Request(poll_url, headers={"User-Agent": "MaaSitalaStudioV8/1.0"})
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            raw_lines = resp.read().decode("utf-8", errors="ignore").strip().splitlines()
        if not raw_lines:
            return None

        # Group attachments by shot_id (supporting MSV8 multi-chunk, MSV7 1of1, and legacy shoot.bin)
        shot_groups: dict[str, dict] = {}
        shot_order: list[str] = []
        for ln in reversed(raw_lines):
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            att = obj.get("attachment")
            if not isinstance(att, dict) or not att.get("url"):
                continue
            att_url = str(att["url"])
            att_name = str(att.get("name") or obj.get("filename") or "shoot.bin")
            msg_ts = float(obj.get("time") or now)

            m = _MSV_FNAME_RE.match(att_name)
            if m:
                proto, shot_hex, idx_s, tot_s = m.group(1), m.group(2).lower(), int(m.group(3)), int(m.group(4))
                skey = f"{proto}_{shot_hex}"
                idx0 = max(0, idx_s - 1)
                tot_n = max(1, tot_s)
            else:
                msg_id = str(obj.get("id") or f"{msg_ts:.3f}")
                skey = f"legacy_{msg_id}"
                idx0 = 0
                tot_n = 1

            if skey not in shot_groups:
                shot_groups[skey] = {"total": tot_n, "ts": msg_ts, "urls": {}}
                shot_order.append(skey)
            grp = shot_groups[skey]
            grp["ts"] = max(float(grp["ts"]), msg_ts)
            if idx0 not in grp["urls"]:
                grp["urls"][idx0] = att_url

        # Pick the newest COMPLETE shot (all 0..total-1 chunks uploaded!)
        chosen_key = ""
        chosen_grp = None
        for skey in shot_order:
            grp = shot_groups[skey]
            tot_n = int(grp["total"])
            if all(i in grp["urls"] for i in range(tot_n)):
                chosen_key = skey
                chosen_grp = grp
                break

        if not chosen_grp or not chosen_key:
            return None

        dest_path = os.path.abspath(os.path.join(MOBILE_SHOOT_DIR, f"{target_sid}.png"))
        latest_ts = float(chosen_grp["ts"])
        with _STORE_LOCK:
            prev_key = _REMOTE_SHOOT_KEYS.get(target_sid, "")
            prev_remote = _REMOTE_SHOOT_UPDATED.get(target_sid, 0.0)
        if os.path.isfile(dest_path) and (chosen_key == prev_key or (chosen_key.startswith("legacy_") and latest_ts <= prev_remote)):
            return None

        tot_n = int(chosen_grp["total"])
        raw_packets = [_download_relay_attachment_cached(chosen_grp["urls"][i]) for i in range(tot_n)]
        plain_img_bytes = reassemble_msv8_or_msv7_packets(raw_packets, aes_key)

        with Image.open(io.BytesIO(plain_img_bytes)) as im:
            im_rgb = ImageOps.exif_transpose(im).convert("RGB")
            rgb_arr = np.array(im_rgb, dtype=np.uint8)

        os.makedirs(MOBILE_SHOOT_DIR, exist_ok=True)
        imwrite_rgb(dest_path, rgb_arr)
        # Ensure filesystem mtime advances monotonically even for rapid back-to-back shots
        touch_ts = max(time.time(), prev_remote + 0.01)
        try:
            os.utime(dest_path, (touch_ts, touch_ts))
        except Exception:
            pass

        with _STORE_LOCK:
            _REMOTE_SHOOT_KEYS[target_sid] = chosen_key
            _REMOTE_SHOOT_UPDATED[target_sid] = max(latest_ts, touch_ts)
            entry = _MOBILE_SHOOT_STORE.get(target_sid, {"session_id": target_sid, "slot_path": dest_path})
            entry["photo_path"] = dest_path
            entry["updated_at"] = touch_ts
            _MOBILE_SHOOT_STORE[target_sid] = entry
        return dest_path
    except Exception:
        return None


def poll_mobile_shoot_photo_status(
    session_id: str | None = None,
    only_new: bool = False,
) -> tuple[np.ndarray | None, str]:
    """
    Check if the phone has uploaded a photo for `session_id` (or latest session) — checking both
    the Zero-HF-Login Edge Gateway (`Maa-Sitala-Portal`) and local storage — and return `(img_rgb, status_msg)`.
    When `only_new=True` (used by background `gr.Timer` polling), returns `(img_rgb, msg)` only once per new upload.
    """
    os.makedirs(MOBILE_SHOOT_DIR, exist_ok=True)
    candidate_path = None
    used_sid = ""

    with _STORE_LOCK:
        target_sid = _sanitize_token(session_id) if session_id and str(session_id).strip() else _LATEST_MOBILE_SESSION_ID
        if target_sid and target_sid in _MOBILE_SHOOT_STORE:
            candidate_path = _MOBILE_SHOOT_STORE[target_sid].get("photo_path")
            used_sid = target_sid

    if target_sid:
        pulled_path = _try_pull_shoot_from_portal(target_sid, force=not only_new)
        if pulled_path and os.path.isfile(pulled_path):
            candidate_path = pulled_path
            used_sid = target_sid

    if not candidate_path and session_id and str(session_id).strip():
        clean_sid = _sanitize_token(session_id)
        p = os.path.join(MOBILE_SHOOT_DIR, f"{clean_sid}.png")
        if os.path.isfile(p):
            candidate_path = p
            used_sid = clean_sid

    if not candidate_path and not (session_id and str(session_id).strip()):
        pngs = [
            os.path.join(MOBILE_SHOOT_DIR, f)
            for f in os.listdir(MOBILE_SHOOT_DIR)
            if f.lower().endswith(".png")
        ]
        if pngs:
            pngs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            candidate_path = pngs[0]
            used_sid = os.path.splitext(os.path.basename(candidate_path))[0]

    if candidate_path and os.path.isfile(candidate_path):
        mtime = float(os.path.getmtime(candidate_path))
        if only_new:
            with _STORE_LOCK:
                last_seen = _CONSUMED_MOBILE_MTIMES.get(used_sid, 0.0)
                if mtime <= last_seen:
                    return None, ""
                _CONSUMED_MOBILE_MTIMES[used_sid] = mtime
        img = imread_rgb(candidate_path)
        if img is not None:
            h, w = img.shape[:2]
            return img, f"✅ Loaded phone camera photo (`{used_sid}`, {w}×{h}px) into Studio! Click **🚀 Run** to process."
    return None, "ℹ️ Waiting for phone camera upload... Scan the QR code on your phone (no HF login required), snap a photo, and tap **Send to Studio**."


def poll_mobile_shoot_photo(session_id: str | None = None) -> np.ndarray | None:
    """
    Check if the phone has uploaded a photo for `session_id` and return the RGB `np.ndarray` (or `None`).
    """
    img, _ = poll_mobile_shoot_photo_status(session_id=session_id, only_new=False)
    return img


# ============================================================================
# 4. PERMANENT CUSTOMER SHARE LINK & 4-DIGIT PIN VAULT (?share=<share_id>)
# ============================================================================

def _publish_encrypted_share_to_portal(share_id: str, meta_bytes: bytes, enc_bytes: bytes) -> bool:
    """
    Commit `vault/{share_id}.meta.json` and `vault/{share_id}.enc` (`AES-256-GCM` ciphertext)
    atomically to the public Static Space `PUBLIC_PORTAL_SPACE_ID` (`DevAnupam2008/Maa-Sitala-Portal`)
    so any customer can open `share.html?id={share_id}#k=...` with zero Hugging Face login 24/7/365.
    """
    if not HF_TOKEN or not PUBLIC_PORTAL_SPACE_ID or _is_offline_test_mode():
        return False

    if HAS_HF_HUB and HfApi is not None and CommitOperationAdd is not None:
        try:
            api = HfApi(token=HF_TOKEN)
            api.create_commit(
                repo_id=PUBLIC_PORTAL_SPACE_ID,
                repo_type="space",
                operations=[
                    CommitOperationAdd(path_in_repo=f"vault/{share_id}.meta.json", path_or_fileobj=meta_bytes),
                    CommitOperationAdd(path_in_repo=f"vault/{share_id}.enc", path_or_fileobj=enc_bytes),
                ],
                commit_message=f"Add encrypted share vault {share_id}",
            )
            return True
        except Exception:
            pass

    # Standard-library NDJSON commit fallback (zero extra dependencies)
    try:
        lines = [
            json.dumps({"key": "header", "value": {"summary": f"Add encrypted share vault {share_id}", "description": ""}}),
            json.dumps({
                "key": "file",
                "value": {
                    "content": base64.b64encode(meta_bytes).decode("ascii"),
                    "path": f"vault/{share_id}.meta.json",
                    "encoding": "base64",
                },
            }),
            json.dumps({
                "key": "file",
                "value": {
                    "content": base64.b64encode(enc_bytes).decode("ascii"),
                    "path": f"vault/{share_id}.enc",
                    "encoding": "base64",
                },
            }),
        ]
        ndjson_body = ("\n".join(lines) + "\n").encode("utf-8")
        req = urllib.request.Request(
            f"https://huggingface.co/api/spaces/{PUBLIC_PORTAL_SPACE_ID}/commit/main",
            data=ndjson_body,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/x-ndjson",
                "User-Agent": "MaaSitalaStudioV7/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.status in (200, 201)
    except Exception:
        return False


def _load_shares_index() -> dict:
    for path in (SHARES_INDEX_PATH, os.path.join(DB_DIR, "shares_index.json")):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
    return {}


def _save_shares_index(index_data: dict) -> None:
    os.makedirs(SHARES_DIR, exist_ok=True)
    with open(SHARES_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)


def _try_pull_share_from_cloud(share_id: str) -> None:
    """
    If `{share_id}.png` or `shares_index.json` is not present locally in `shop_data/database/shares/`,
    pull it on-demand from `DevAnupam2008/Studio-Database` via `hf_hub_download` or `urllib.request`.
    """
    if not REPO_ID or os.environ.get("PASSPORT_OFFLINE_TESTS", "0").strip() in ("1", "true", "TRUE", "yes"):
        return
    os.makedirs(SHARES_DIR, exist_ok=True)
    rel_files = ["shares/shares_index.json", f"shares/{share_id}.png"]
    for rel_f in rel_files:
        local_target = os.path.join(DB_DIR, rel_f)
        if os.path.isfile(local_target) and os.path.getsize(local_target) > 0:
            continue
        if HAS_HF_HUB and hf_hub_download is not None:
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=rel_f,
                    token=HF_TOKEN,
                    local_dir=DB_DIR,
                )
                if os.path.isfile(local_target):
                    continue
            except Exception:
                pass
        try:
            import urllib.request
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{rel_f}"
            headers = {"User-Agent": "PassportStudioV6/1.0"}
            if HF_TOKEN:
                headers["Authorization"] = f"Bearer {HF_TOKEN}"
            req = urllib.request.Request(url, headers=headers)
            os.makedirs(os.path.dirname(local_target), exist_ok=True)
            with urllib.request.urlopen(req, timeout=15) as resp, open(local_target, "wb") as out_f:
                out_f.write(resp.read())
        except Exception:
            pass


def create_customer_share(
    img_rgb: np.ndarray | None = None,
    customer_name: str = "Studio Customer",
    custom_pin: str = "",
    base_url: str = "",
    *,
    photo_rgb: np.ndarray | None = None,
    sheet_rgb: np.ndarray | None = None,
    pin_code: str = "",
    request=None,
) -> dict:
    """
    Create a permanent, 4-Digit PIN-protected Customer Share Vault entry (`?share=<share_id>`)
    synced to SQLite (`customer_shares`), `shop_data/database/shares/{share_id}.png`,
    `shop_data/database/shares/shares_index.json`, and Hugging Face Dataset `DevAnupam2008/Studio-Database`.
    """
    target_rgb = img_rgb if img_rgb is not None else photo_rgb
    if target_rgb is None or not isinstance(target_rgb, np.ndarray):
        raise ValueError("img_rgb / photo_rgb is required to create a share link")

    init_db()
    os.makedirs(SHARES_DIR, exist_ok=True)

    # Normalize or auto-generate 4-digit numeric PIN (1000..9999)
    raw_pin = custom_pin if str(custom_pin or "").strip() else pin_code
    digits = re.sub(r"\D", "", str(raw_pin or "").strip())
    if len(digits) != 4:
        digits = f"{secrets.randbelow(9000) + 1000:04d}"

    share_id = f"ms-{secrets.token_hex(3)}"
    clean_name = (str(customer_name or "").strip() or "Studio Customer")[:80]

    pin_salt = secrets.token_hex(8)
    pin_sha256 = hashlib.sha256((pin_salt + digits).encode("utf-8")).hexdigest()

    photo_path = os.path.abspath(os.path.join(SHARES_DIR, f"{share_id}.png"))
    imwrite_rgb(photo_path, target_rgb[:, :, :3])

    sheet_path = ""
    if sheet_rgb is not None and isinstance(sheet_rgb, np.ndarray):
        sheet_path = os.path.abspath(os.path.join(SHARES_DIR, f"{share_id}_sheet.png"))
        imwrite_rgb(sheet_path, sheet_rgb[:, :, :3])

    h, w = target_rgb.shape[:2]
    resolution_str = f"{w} × {h} px"
    created_at = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO customer_shares
        (share_id, customer_name, pin_salt, pin_sha256, pin_hash, photo_path, sheet_path, resolution_str, width, height, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (share_id, clean_name, pin_salt, pin_sha256, pin_sha256, photo_path, sheet_path, resolution_str, int(w), int(h), created_at),
    )
    conn.commit()
    conn.close()

    idx = _load_shares_index()
    idx[share_id] = {
        "share_id": share_id,
        "customer_name": clean_name,
        "pin_salt": pin_salt,
        "pin_sha256": pin_sha256,
        "pin_hash": pin_sha256,
        "photo_rel": f"shares/{share_id}.png",
        "sheet_rel": f"shares/{share_id}_sheet.png" if sheet_path else "",
        "resolution_str": resolution_str,
        "width": int(w),
        "height": int(h),
        "created_at": created_at,
    }
    _save_shares_index(idx)

    # Sync to Hugging Face Dataset (DevAnupam2008/Studio-Database) in background when not running offline unit tests
    if not _is_offline_test_mode():
        threading.Thread(target=push_to_cloud, daemon=True).start()

    if base_url and str(base_url).strip():
        resolved_base = str(base_url).strip().rstrip("/")
        share_url = f"{resolved_base}/?share={share_id}"
    elif PUBLIC_PORTAL_BASE_URL:
        portal_base = PUBLIC_PORTAL_BASE_URL.rstrip("/")
        url_k = secrets.token_hex(12)
        share_url = f"{portal_base}/share.html?id={share_id}#k={url_k}"
        if not _is_offline_test_mode() and HAS_AESGCM:
            try:
                kdf_salt = secrets.token_bytes(16)
                iv12 = secrets.token_bytes(12)
                iters = 150_000
                with open(photo_path, "rb") as pf:
                    png_bytes = pf.read()
                aes_key = hashlib.pbkdf2_hmac(
                    "sha256",
                    f"{url_k}:{digits}".encode("utf-8"),
                    kdf_salt,
                    iters,
                    dklen=32,
                )
                enc_bytes = AESGCM(aes_key).encrypt(iv12, png_bytes, share_id.encode("utf-8"))
                vault_meta = {
                    "v": 7,
                    "share_id": share_id,
                    "customer_name": clean_name,
                    "resolution_str": resolution_str,
                    "width": int(w),
                    "height": int(h),
                    "salt_hex": kdf_salt.hex(),
                    "iv_hex": iv12.hex(),
                    "iter": iters,
                    "created_at": created_at,
                }
                meta_bytes = json.dumps(vault_meta, indent=2).encode("utf-8")
                _publish_encrypted_share_to_portal(share_id, meta_bytes, enc_bytes)
            except Exception:
                pass
    else:
        resolved_base = get_public_studio_base_url(request)
        share_url = f"{resolved_base}/?share={share_id}"

    qr_rgb = generate_qr_image_rgb(share_url, size=280, border=4)
    qr_svg_uri = generate_qr_svg_data_uri(share_url, border=3)

    return {
        "share_id": share_id,
        "customer_name": clean_name,
        "pin_code": digits,
        "pin": digits,
        "share_url": share_url,
        "qr_rgb": qr_rgb,
        "qr_svg_uri": qr_svg_uri,
        "photo_path": photo_path,
        "sheet_path": sheet_path,
        "resolution_str": resolution_str,
        "width": int(w),
        "height": int(h),
    }


def get_customer_share_meta(share_id: str) -> dict | None:
    """
    Retrieve public metadata for a `share_id` (never returns the raw PIN or hash).
    """
    init_db()
    clean_id = _sanitize_token(share_id, prefix="ms")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT share_id, customer_name, resolution_str, width, height, created_at FROM customer_shares WHERE share_id = ?",
        (clean_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "share_id": row[0],
            "customer_name": row[1],
            "resolution_str": row[2],
            "width": int(row[3] or 0),
            "height": int(row[4] or 0),
            "created_at": row[5],
        }
    idx = _load_shares_index()
    if clean_id not in idx:
        _try_pull_share_from_cloud(clean_id)
        idx = _load_shares_index()
    if clean_id in idx:
        rec = idx[clean_id]
        return {
            "share_id": clean_id,
            "customer_name": rec.get("customer_name", "Studio Customer"),
            "resolution_str": rec.get("resolution_str", ""),
            "width": int(rec.get("width", 0)),
            "height": int(rec.get("height", 0)),
            "created_at": rec.get("created_at", 0.0),
        }
    return None


def render_interactive_deep_zoom_html(
    img_rgb: np.ndarray,
    share_id: str = "",
    customer_name: str = "Studio Customer",
) -> str:
    """
    Render a self-contained Interactive 1x–8x Deep-Zoom HTML5 viewer (1x to 8x Zoom Slider +
    Mouse-Wheel Zoom + Touch/Drag Pan + Quick Zoom Buttons) and 1-Click Full-Resolution
    Lossless PNG Download button (`<a download="Maa_Sitala_Passport_{share_id}.png" ...>`).
    """
    h, w = img_rgb.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(img_rgb[:, :, :3], mode="RGB").save(buf, format="PNG")
    b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64_png}"
    safe_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", str(customer_name or "Customer")).strip() or "Customer"
    clean_sid = re.sub(r"[^a-zA-Z0-9_\-]", "", str(share_id or "photo")) or "photo"
    dl_filename = f"Maa_Sitala_Passport_{clean_sid}.png"
    viewer_id = f"dz_{re.sub(r'[^a-zA-Z0-9]', '', clean_sid)[:16]}"

    return f"""
<div style="background:#0f172a;border:1px solid #334155;border-radius:14px;padding:16px;color:#f8fafc;font-family:system-ui,-apple-system,sans-serif;">
  <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px;">
    <div>
      <div style="font-size:16px;font-weight:700;color:#38bdf8;">🔍 Interactive 1x–8x Deep-Zoom Studio Inspector</div>
      <div style="font-size:13px;color:#94a3b8;">Customer: <b style="color:#f1f5f9;">{safe_name}</b> &bull; Native Resolution: <b style="color:#4ade80;">{w} &times; {h} px (Lossless PNG)</b></div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
      <label style="font-size:12px;color:#cbd5e1;display:flex;align-items:center;gap:6px;">
        <span>Zoom: <b id="{viewer_id}_lbl">1.0x</b></span>
        <input id="{viewer_id}_slider" type="range" min="1" max="8" step="0.1" value="1"
          oninput="window['{viewer_id}_zoom'](parseFloat(this.value))"
          style="width:110px;accent-color:#38bdf8;cursor:pointer;" />
      </label>
      <button type="button" onclick="window['{viewer_id}_zoom'](1)" style="background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:6px 10px;cursor:pointer;font-weight:600;">1x Fit</button>
      <button type="button" onclick="window['{viewer_id}_zoom'](2)" style="background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:6px 10px;cursor:pointer;font-weight:600;">2x</button>
      <button type="button" onclick="window['{viewer_id}_zoom'](4)" style="background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:6px 10px;cursor:pointer;font-weight:600;">4x HD</button>
      <button type="button" onclick="window['{viewer_id}_zoom'](8)" style="background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:6px 10px;cursor:pointer;font-weight:600;">8x Pore Zoom</button>
      <a href="{data_uri}" download="{dl_filename}" style="background:linear-gradient(135deg,#0284c7,#2563eb);color:#ffffff;text-decoration:none;border-radius:8px;padding:7px 14px;font-weight:700;display:inline-block;box-shadow:0 2px 8px rgba(37,99,235,0.4);">📥 Download Original PNG ({w}&times;{h})</a>
    </div>
  </div>
  <div id="{viewer_id}_box" style="width:100%;height:460px;overflow:auto;background:#020617;border:1px solid #1e293b;border-radius:10px;display:flex;justify-content:center;align-items:flex-start;padding:12px;cursor:grab;user-select:none;">
    <img id="{viewer_id}_img" src="{data_uri}" draggable="false" alt="Studio Portrait" style="width:360px;max-width:none;height:auto;border-radius:6px;transition:width 0.12s ease-out;box-shadow:0 6px 24px rgba(0,0,0,0.65);" onclick="window['{viewer_id}_cycle']()" />
  </div>
  <script>
    (function() {{
      var baseW = 360;
      var curScale = 1.0;
      var box = document.getElementById('{viewer_id}_box');
      var img = document.getElementById('{viewer_id}_img');
      var slider = document.getElementById('{viewer_id}_slider');
      var lbl = document.getElementById('{viewer_id}_lbl');
      var isDown = false, startX = 0, startY = 0, scrollL = 0, scrollT = 0;

      window['{viewer_id}_zoom'] = function(s) {{
        curScale = Math.max(1.0, Math.min(8.0, parseFloat(s) || 1.0));
        if (img) {{ img.style.width = Math.round(baseW * curScale) + 'px'; }}
        if (slider) {{ slider.value = curScale; }}
        if (lbl) {{ lbl.textContent = curScale.toFixed(1) + 'x'; }}
      }};
      window['{viewer_id}_cycle'] = function() {{
        var next = curScale < 1.9 ? 2 : (curScale < 3.9 ? 4 : (curScale < 7.9 ? 8 : 1));
        window['{viewer_id}_zoom'](next);
      }};
      if (box) {{
        box.addEventListener('wheel', function(e) {{
          e.preventDefault();
          var delta = e.deltaY < 0 ? 0.4 : -0.4;
          window['{viewer_id}_zoom'](curScale + delta);
        }}, {{passive: false}});
        box.addEventListener('mousedown', function(e) {{
          isDown = true;
          box.style.cursor = 'grabbing';
          startX = e.pageX - box.offsetLeft;
          startY = e.pageY - box.offsetTop;
          scrollL = box.scrollLeft;
          scrollT = box.scrollTop;
        }});
        window.addEventListener('mouseup', function() {{
          isDown = false;
          if (box) box.style.cursor = 'grab';
        }});
        box.addEventListener('mousemove', function(e) {{
          if (!isDown) return;
          e.preventDefault();
          var x = e.pageX - box.offsetLeft;
          var y = e.pageY - box.offsetTop;
          box.scrollLeft = scrollL - (x - startX);
          box.scrollTop = scrollT - (y - startY);
        }});
      }}
    }})();
  </script>
</div>
"""


def verify_customer_share_pin(
    share_id: str,
    pin_code: str = "",
    *,
    entered_pin: str = "",
) -> tuple[bool, np.ndarray | None, str | None, dict]:
    """
    Verify the 4-digit PIN for `share_id` using constant-time `hmac.compare_digest`.
    If `{share_id}.png` is not in local `shop_data/database/shares/`, pulls it on-demand from
    `DevAnupam2008/Studio-Database` (`hf_hub_download` / `urllib.request`).
    Returns `(True, full_res_rgb, png_file_path, meta)` on valid 4-digit PIN,
    or `(False, None, None, {"error": ...})` on invalid PIN.
    """
    init_db()
    clean_id = _sanitize_token(share_id, prefix="ms")
    raw_pin = pin_code if str(pin_code or "").strip() else entered_pin
    clean_pin = re.sub(r"\D", "", str(raw_pin or "").strip())
    if len(clean_pin) != 4:
        err_msg = "❌ Please enter a valid 4-digit numeric PIN."
        return False, None, None, {"error": err_msg, "message": err_msg}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT customer_name, pin_salt, pin_sha256, pin_hash, photo_path, sheet_path, resolution_str, width, height, created_at FROM customer_shares WHERE share_id = ?",
        (clean_id,),
    )
    row = cursor.fetchone()
    conn.close()

    rec = None
    if row:
        rec = {
            "customer_name": row[0],
            "pin_salt": row[1],
            "pin_sha256": row[2] or row[3] or "",
            "photo_path": row[4],
            "sheet_path": row[5],
            "resolution_str": row[6],
            "width": int(row[7] or 0),
            "height": int(row[8] or 0),
            "created_at": row[9],
        }
    else:
        idx = _load_shares_index()
        if clean_id not in idx:
            _try_pull_share_from_cloud(clean_id)
            idx = _load_shares_index()
        if clean_id in idx:
            item = idx[clean_id]
            rec = {
                "customer_name": item.get("customer_name", "Studio Customer"),
                "pin_salt": item.get("pin_salt", ""),
                "pin_sha256": item.get("pin_sha256") or item.get("pin_hash") or "",
                "photo_path": os.path.join(DB_DIR, item.get("photo_rel", f"shares/{clean_id}.png")),
                "sheet_path": os.path.join(DB_DIR, item.get("sheet_rel", "")) if item.get("sheet_rel") else "",
                "resolution_str": item.get("resolution_str", ""),
                "width": int(item.get("width", 0)),
                "height": int(item.get("height", 0)),
                "created_at": item.get("created_at", 0.0),
            }

    if rec is None:
        err_msg = f"❌ Share ID `{clean_id}` not found."
        return False, None, None, {"error": err_msg, "message": err_msg}

    cand_no_colon = hashlib.sha256((str(rec["pin_salt"]) + clean_pin).encode("utf-8")).hexdigest()
    cand_with_colon = hashlib.sha256(f"{rec['pin_salt']}:{clean_pin}".encode("utf-8")).hexdigest()
    stored_hash = str(rec["pin_sha256"])
    pin_ok = hmac.compare_digest(stored_hash, cand_no_colon) or hmac.compare_digest(stored_hash, cand_with_colon)
    if not pin_ok:
        err_msg = "❌ Incorrect 4-Digit PIN Code. Access Denied."
        return False, None, None, {"error": err_msg, "message": err_msg}

    photo_path = rec["photo_path"]
    if not os.path.isfile(photo_path):
        alt_path = os.path.join(SHARES_DIR, f"{clean_id}.png")
        if not os.path.isfile(alt_path):
            _try_pull_share_from_cloud(clean_id)
        if os.path.isfile(alt_path):
            photo_path = alt_path
        else:
            err_msg = "❌ Shared photo file is missing on disk."
            return False, None, None, {"error": err_msg, "message": err_msg}

    photo_rgb = imread_rgb(photo_path)
    if photo_rgb is None:
        err_msg = "❌ Could not decode shared photo file."
        return False, None, None, {"error": err_msg, "message": err_msg}

    sheet_path = rec.get("sheet_path") or ""
    sheet_rgb = imread_rgb(sheet_path) if (sheet_path and os.path.isfile(sheet_path)) else None

    deep_zoom_html = render_interactive_deep_zoom_html(
        photo_rgb,
        share_id=clean_id,
        customer_name=rec["customer_name"],
    )
    ok_msg = f"✅ PIN Verified! Unlocked **{rec['customer_name']}** ({rec['resolution_str']})."
    meta = {
        "share_id": clean_id,
        "customer_name": rec["customer_name"],
        "photo_rgb": photo_rgb,
        "sheet_rgb": sheet_rgb,
        "photo_path": photo_path,
        "sheet_path": sheet_path if sheet_rgb is not None else None,
        "resolution_str": rec["resolution_str"],
        "width": int(photo_rgb.shape[1]),
        "height": int(photo_rgb.shape[0]),
        "deep_zoom_html": deep_zoom_html,
        "message": ok_msg,
    }
    return True, photo_rgb, photo_path, meta
