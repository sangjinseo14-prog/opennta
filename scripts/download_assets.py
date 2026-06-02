"""Download large binary assets from a GitHub Release.

These files are intentionally not tracked in git (see ``.gitignore``).
Run this once after cloning the repository::

    python download_assets.py                 # download if missing
    python download_assets.py --force         # re-download even if files already exist
    python download_assets.py --tag assets_v1.0.0 # pin to a specific release tag

The script only uses the Python standard library, so it can be executed
before installing project dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "sangjinseo14-prog/opennta"
DEFAULT_TAG = "assets_v1.0.0"
ROOT = Path(__file__).resolve().parent.parent

# 1 MiB streaming chunk: large enough to keep urllib/disk syscalls cheap,
# small enough to print smooth progress on slow links.
DOWNLOAD_CHUNK_BYTES = 1 << 20

ASSET_FILENAME = "unet_assets.zip"
# SHA-256 of the zip itself. Set to None to skip verification.
ASSET_SHA256: str | None = "c8ec2d9518efbed4157d918039c726d8364fef8fbf141a2444aabd432a06fb1c"

ASSET_TARGET_DIR = ROOT / "src" / "opennta" / "analysis" / "unet_field" / "unet_bundle" / "unet_assets"
# Every file the zip is expected to deliver. If all are present, skip the download.
ASSET_FILES: tuple[str, ...] = (
    "best.weights.h5",
    "meta.csv",
    "norm.json",
    "reference_field.npz",
)

# Per-file SHA-256 of each extracted asset, verified after extraction. Leave a
# value as None to skip that file; fill in real hashes after a trusted upload to
# harden integrity beyond the single zip-level checksum.
ASSET_FILE_SHA256: dict[str, str | None] = {
    "best.weights.h5": "a219edcae9c74532cf72ba6e717832f405cecfdbfc1059261d4c9522b50d4c55",
    "meta.csv": "c94b7516420e645c1e3e3309407a341f01824b0c429175fd77506ed9918e959c",
    "norm.json": "21795a1f2239dcdf136d248534cac70941856c86c195688b490de4a7e191d17d",
    "reference_field.npz": "b3503793b57e93b9a475509c38bbde04abda3cae3b13306f67a65aff36a1d209",
}


def release_url(tag: str, filename: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{filename}"


def sha256_of(path: Path, chunk: int = DOWNLOAD_CHUNK_BYTES) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> {url}")

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        with tmp.open("wb") as out:
            while True:
                buf = resp.read(DOWNLOAD_CHUNK_BYTES)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded * 100 // total
                    if pct != last_pct and pct % 5 == 0:
                        print(f"     {pct:3d}%  ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)")
                        last_pct = pct
    tmp.replace(dest)


def _strip_common_prefix(names: list[str]) -> str:
    # If every entry sits under a single top-level directory, return that
    # prefix (with trailing slash) so callers can strip it during extraction.
    names = [n for n in names if n and not n.startswith("__MACOSX/")]
    if not names:
        return ""
    first = names[0].split("/", 1)[0]
    if not first:
        return ""
    prefix = first + "/"
    if all(n == first or n.startswith(prefix) for n in names):
        return prefix
    return ""


def extract_zip(zip_path: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = target_dir.resolve()
    written: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        prefix = _strip_common_prefix([m.filename for m in members])

        for member in members:
            name = member.filename
            if name.startswith("__MACOSX/") or name.endswith("/.DS_Store"):
                continue
            if prefix and name.startswith(prefix):
                name = name[len(prefix):]
            if not name:
                continue

            out_path = (resolved_target / name).resolve()
            # Reject paths that would escape the target directory.
            if resolved_target not in out_path.parents and out_path != resolved_target:
                raise RuntimeError(f"unsafe path in zip: {member.filename}")

            if member.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=DOWNLOAD_CHUNK_BYTES)
            written.append(name)
    return written


def verify_extracted_files() -> list[str]:
    # Returns a list of mismatch messages (empty if all good). Files whose
    # pinned hash is None are skipped (not yet pinned).
    errors: list[str] = []
    for name, expected in ASSET_FILE_SHA256.items():
        if expected is None:
            continue
        path = ASSET_TARGET_DIR / name
        if not path.exists():
            errors.append(f"{name}: missing")
            continue
        actual = sha256_of(path)
        if actual.lower() != expected.lower():
            errors.append(f"{name}: expected {expected}, got {actual}")
    return errors


def ensure_assets(tag: str, force: bool) -> bool:
    if not force:
        missing = [n for n in ASSET_FILES if not (ASSET_TARGET_DIR / n).exists()]
        if not missing:
            print(f"[skip] {ASSET_TARGET_DIR.relative_to(ROOT)} (all {len(ASSET_FILES)} files present)")
            return True

    url = release_url(tag, ASSET_FILENAME)
    print(f"[get ] {ASSET_FILENAME}")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / ASSET_FILENAME
        try:
            download(url, zip_path)
        except urllib.error.HTTPError as e:
            print(f"[fail] HTTP {e.code} for {url}", file=sys.stderr)
            return False
        except urllib.error.URLError as e:
            print(f"[fail] {e.reason} for {url}", file=sys.stderr)
            return False

        if ASSET_SHA256:
            actual = sha256_of(zip_path)
            if actual.lower() != ASSET_SHA256.lower():
                print(
                    f"[fail] checksum mismatch for {ASSET_FILENAME}: "
                    f"expected {ASSET_SHA256}, got {actual}",
                    file=sys.stderr,
                )
                return False
            print("[ ok ] checksum verified")

        print(f"[ext ] -> {ASSET_TARGET_DIR.relative_to(ROOT)}")
        try:
            written = extract_zip(zip_path, ASSET_TARGET_DIR)
        except zipfile.BadZipFile as e:
            print(f"[fail] {ASSET_FILENAME} is not a valid zip: {e}", file=sys.stderr)
            return False
        except RuntimeError as e:
            print(f"[fail] {e}", file=sys.stderr)
            return False

        for name in written:
            print(f"       {name}")

    missing = [n for n in ASSET_FILES if not (ASSET_TARGET_DIR / n).exists()]
    if missing:
        print(
            f"[fail] {ASSET_FILENAME} is missing expected file(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return False

    file_errors = verify_extracted_files()
    if file_errors:
        for err in file_errors:
            print(f"[fail] file checksum: {err}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OpenNTA release assets.")
    parser.add_argument("--tag", default=os.environ.get("OPENNTA_ASSETS_TAG", DEFAULT_TAG),
                        help=f"GitHub release tag to download from (default: {DEFAULT_TAG})")
    parser.add_argument("--force", action="store_true",
                        help="redownload and re-extract even if the assets are already present")
    args = parser.parse_args(argv)

    print(f"Repository : {REPO}")
    print(f"Release tag: {args.tag}")
    print(f"Asset      : {ASSET_FILENAME}")
    print()

    ok = ensure_assets(args.tag, args.force)
    print()
    if not ok:
        print("Done with 1 failure(s).", file=sys.stderr)
        return 1
    print("All assets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
