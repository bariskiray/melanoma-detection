"""Download HAM10000 (and optionally the ISIC 2018 Task 3 test set) from Harvard Dataverse.

Images are resized on the fly so that the shorter side is ``--size`` pixels and the
original zip archives are deleted afterwards. This keeps the on-disk footprint around
250 MB instead of ~3 GB, which is plenty for training at 224x224.

Usage:
    python scripts/download_data.py --out data            # HAM10000 only
    python scripts/download_data.py --out data --isic2018 # + external test set
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{id}"

HAM_FILES = {
    "HAM10000_metadata.csv": (4338392, "original"),
    "HAM10000_images_part_1.zip": (3172585, None),
    "HAM10000_images_part_2.zip": (3172584, None),
}
ISIC2018_FILES = {
    "ISIC2018_Task3_Test_GroundTruth.csv": (6924466, "original"),
    "ISIC2018_Task3_Test_Images.zip": (3855824, None),
}


HEADERS = {"User-Agent": "Mozilla/5.0 (dataset-downloader)"}  # Dataverse 403s urllib's default UA
CHUNK = 16 << 20


def _get(url: str, start: int | None = None, end: int | None = None):
    headers = dict(HEADERS)
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def download(file_id: int, dest: Path, fmt: str | None, workers: int = 8) -> None:
    """Download with parallel HTTP range requests.

    Dataverse redirects to S3, which throttles a single long-lived connection (~1 MB/s);
    several concurrent range requests are typically 4-8x faster.
    """
    url = DATAVERSE.format(id=file_id) + (f"?format={fmt}" if fmt else "")
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...", flush=True)
    with _get(url, 0, 0) as probe:  # follow the redirect once and learn the size
        s3_url = probe.geturl()
        content_range = probe.headers.get("Content-Range")
    if probe.status != 206 or not content_range:
        with _get(url) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, CHUNK)
        tmp.rename(dest)
        return

    total = int(content_range.split("/")[-1])
    with open(tmp, "wb") as fh:
        fh.truncate(total)
    ranges = [(s, min(s + CHUNK, total) - 1) for s in range(0, total, CHUNK)]
    done = 0

    def fetch_range(r: tuple[int, int]) -> int:
        for attempt in range(5):
            try:
                with _get(s3_url, *r) as resp:
                    data = resp.read()
                if len(data) != r[1] - r[0] + 1:
                    raise OSError("short read")
                with open(tmp, "r+b") as fh:
                    fh.seek(r[0])
                    fh.write(data)
                return len(data)
            except OSError:
                if attempt == 4:
                    raise
        return 0

    with ThreadPoolExecutor(workers) as pool:
        for n in pool.map(fetch_range, ranges):
            done += n
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def extract_resized(zip_path: Path, out_dir: Path, size: int) -> int:
    """Extract every JPEG in ``zip_path`` into ``out_dir`` with shorter side == size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            # skip directories and macOS resource-fork entries (__MACOSX/._*.jpg)
            if info.is_dir() or not name.lower().endswith(".jpg") or name.startswith("._"):
                continue
            if "__MACOSX" in info.filename:
                continue
            with zf.open(info) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            w, h = img.size
            scale = size / min(w, h)
            if scale < 1:
                img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            img.save(out_dir / name, quality=95)
            n += 1
    return n


def fetch(files: dict, raw_dir: Path, img_dir: Path, size: int) -> None:
    for name, (file_id, fmt) in files.items():
        dest = raw_dir / name
        if name.endswith(".csv"):
            if not dest.exists():
                download(file_id, dest, fmt)
            continue
        marker = raw_dir / f".{name}.done"
        if marker.exists():
            print(f"{name}: already extracted, skipping")
            continue
        download(file_id, dest, fmt)
        n = extract_resized(dest, img_dir, size)
        dest.unlink()
        marker.write_text(f"{n}\n")
        print(f"  extracted {n} images -> {img_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--size", type=int, default=256, help="shorter side after resizing")
    p.add_argument("--isic2018", action="store_true", help="also fetch the ISIC 2018 test set")
    p.add_argument("--skip-ham", action="store_true")
    args = p.parse_args()

    if shutil.disk_usage(args.out.parent if args.out.parent.exists() else ".").free < 2e9:
        print("warning: less than 2 GB free disk space", file=sys.stderr)

    if not args.skip_ham:
        ham = args.out / "ham10000"
        ham.mkdir(parents=True, exist_ok=True)
        fetch(HAM_FILES, ham, ham / "images", args.size)
    if args.isic2018:
        isic = args.out / "isic2018_test"
        isic.mkdir(parents=True, exist_ok=True)
        fetch(ISIC2018_FILES, isic, isic / "images", args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
