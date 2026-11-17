"""Dataset scanning utilities (Phases 2-3)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from PIL import Image


def scan_image(path: Path) -> dict:
    """Read one image's HEADER (cheap) and return its metadata.
    Never raises: failures are recorded in the 'error' column."""
    record = {
        "path": str(path), "label": path.parent.name, "extension": path.suffix.lower(),
        "file_size_kb": path.stat().st_size / 1024,
        "width": None, "height": None, "mode": None, "channels": None,
        "pil_format": None, "error": None,
    }
    try:
        with Image.open(path) as img:   # lazy: reads the header, does not decode pixels
            record.update(width=img.width, height=img.height,
                          mode=img.mode, pil_format=img.format)
            record["channels"] = len(img.getbands())
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return record


def build_metadata(dataset_dir: Path, n_workers: int = 8) -> pd.DataFrame:
    """Recursively scan dataset_dir and return one row per file."""
    files = sorted(p for p in Path(dataset_dir).rglob("*") if p.is_file())
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        rows = list(pool.map(scan_image, files, chunksize=64))
    return pd.DataFrame(rows)


# ---------------- Phase 3: integrity, duplicates ----------------
import hashlib
import numpy as np
import imagehash
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def inspect_image(path) -> dict:
    """FULLY decode one image. Returns integrity status, content hashes and
    32x32 grayscale statistics. Never raises: failures are recorded, not hidden."""
    rec = {"path": str(path), "decode_ok": False, "decode_error": None,
           "md5": None, "phash": None, "thumb_std": None, "thumb_mean": None}
    try:
        rec["md5"] = hashlib.md5(Path(path).read_bytes()).hexdigest()   # exact-duplicate fingerprint
        with Image.open(path) as img:
            img.load()                       # force the full pixel decode (catches truncated files)
            rgb = img.convert("RGB")         # standardise P / L / CMYK / RGBA -> RGB
        rec["phash"] = str(imagehash.phash(rgb))   # perceptual fingerprint (near-duplicates)
        thumb = np.asarray(rgb.convert("L").resize((32, 32)), dtype=np.float32)
        rec["thumb_std"], rec["thumb_mean"] = float(thumb.std()), float(thumb.mean())
        rec["decode_ok"] = True
    except Exception as exc:
        rec["decode_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return rec


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(hex_hashes, max_distance=4, chunk=256):
    """All index pairs (i<j) whose 64-bit perceptual hashes differ in <= max_distance bits."""
    H = np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hex_hashes])
    pairs = []
    for start in range(0, len(H), chunk):
        block = H[start:start + chunk]
        dist = POPCOUNT[block[:, None, :] ^ H[None, :, :]].sum(axis=2, dtype=np.uint16)
        rows, cols = np.nonzero(dist <= max_distance)
        rows = rows + start
        keep = cols > rows
        pairs.append(np.column_stack([rows[keep], cols[keep]]))
    return np.vstack(pairs)


def group_ids_from_pairs(n_items, pairs):
    """Connected components: items linked by near-duplicate pairs share a group id."""
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_items, n_items))
    return connected_components(graph, directed=False)[1]


# ---------------- Phase 4: preprocessing and splitting ----------------
from sklearn.model_selection import StratifiedGroupKFold


def load_rgb(path):
    """Decode ANY supported image and standardise to 3-channel RGB (handles P, L, CMYK, RGBA)."""
    with Image.open(path) as img:
        img.load()
        return img.convert("RGB")


def resize_array(rgb_img, size):
    """Resize (stretching to a square) and return a uint8 array of shape (size, size, 3)."""
    return np.asarray(rgb_img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _load_multi(path, sizes):
    rgb = load_rgb(path)
    return [resize_array(rgb, s) for s in sizes]


def build_caches(paths, sizes=(64, 128, 224), n_workers=8):
    """Decode each image ONCE and fill one uint8 array per size, in the order of `paths`."""
    arrays = [np.empty((len(paths), s, s, 3), dtype=np.uint8) for s in sizes]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, out in enumerate(pool.map(lambda p: _load_multi(p, sizes), paths)):
            for arr, a in zip(arrays, out):
                arr[i] = a
    return arrays


def make_splits(df, seed=42, n_folds=20, n_test_folds=3, n_val_folds=3):
    """70/15/15 split, stratified by label, near-duplicate groups kept together."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label_id"], groups=df["group_id"])):
        fold[idx] = k
    split = np.where(fold < n_test_folds, "test",
            np.where(fold < n_test_folds + n_val_folds, "val", "train"))
    return pd.Series(split, index=df.index, name="split")


# ---------------- Phase 3: integrity, duplicates ----------------
import hashlib
import numpy as np
import imagehash
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def inspect_image(path) -> dict:
    """FULLY decode one image. Returns integrity status, content hashes and
    32x32 grayscale statistics. Never raises: failures are recorded, not hidden."""
    rec = {"path": str(path), "decode_ok": False, "decode_error": None,
           "md5": None, "phash": None, "thumb_std": None, "thumb_mean": None}
    try:
        rec["md5"] = hashlib.md5(Path(path).read_bytes()).hexdigest()   # exact-duplicate fingerprint
        with Image.open(path) as img:
            img.load()                       # force the full pixel decode (catches truncated files)
            rgb = img.convert("RGB")         # standardise P / L / CMYK / RGBA -> RGB
        rec["phash"] = str(imagehash.phash(rgb))   # perceptual fingerprint (near-duplicates)
        thumb = np.asarray(rgb.convert("L").resize((32, 32)), dtype=np.float32)
        rec["thumb_std"], rec["thumb_mean"] = float(thumb.std()), float(thumb.mean())
        rec["decode_ok"] = True
    except Exception as exc:
        rec["decode_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return rec


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(hex_hashes, max_distance=4, chunk=256):
    """All index pairs (i<j) whose 64-bit perceptual hashes differ in <= max_distance bits."""
    H = np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hex_hashes])
    pairs = []
    for start in range(0, len(H), chunk):
        block = H[start:start + chunk]
        dist = POPCOUNT[block[:, None, :] ^ H[None, :, :]].sum(axis=2, dtype=np.uint16)
        rows, cols = np.nonzero(dist <= max_distance)
        rows = rows + start
        keep = cols > rows
        pairs.append(np.column_stack([rows[keep], cols[keep]]))
    return np.vstack(pairs)


def group_ids_from_pairs(n_items, pairs):
    """Connected components: items linked by near-duplicate pairs share a group id."""
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_items, n_items))
    return connected_components(graph, directed=False)[1]


# ---------------- Phase 4: preprocessing and splitting ----------------
from sklearn.model_selection import StratifiedGroupKFold


def load_rgb(path):
    """Decode ANY supported image and standardise to 3-channel RGB (handles P, L, CMYK, RGBA)."""
    with Image.open(path) as img:
        img.load()
        return img.convert("RGB")


def resize_array(rgb_img, size):
    """Resize (stretching to a square) and return a uint8 array of shape (size, size, 3)."""
    return np.asarray(rgb_img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _load_multi(path, sizes):
    rgb = load_rgb(path)
    return [resize_array(rgb, s) for s in sizes]


def build_caches(paths, sizes=(64, 128, 224), n_workers=8):
    """Decode each image ONCE and fill one uint8 array per size, in the order of `paths`."""
    arrays = [np.empty((len(paths), s, s, 3), dtype=np.uint8) for s in sizes]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, out in enumerate(pool.map(lambda p: _load_multi(p, sizes), paths)):
            for arr, a in zip(arrays, out):
                arr[i] = a
    return arrays


def make_splits(df, seed=42, n_folds=20, n_test_folds=3, n_val_folds=3):
    """70/15/15 split, stratified by label, near-duplicate groups kept together."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label_id"], groups=df["group_id"])):
        fold[idx] = k
    split = np.where(fold < n_test_folds, "test",
            np.where(fold < n_test_folds + n_val_folds, "val", "train"))
    return pd.Series(split, index=df.index, name="split")
