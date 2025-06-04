#!/usr/bin/env python3
"""
Hard-coded PDB downloader.
Edit `PDB_IDS`, `OUT_DIR`, `FILE_FORMAT`, and `OVERWRITE` as needed,
then run:  python download_fixed.py
"""

import pathlib
import requests
import sys

# ─── Settings ───────────────────────────────────────────────────────────────────

PDB_IDS = [
    "1A1P", "1A13", "1A1T", "1A24", "1A2I", "1A3P", "1A57", "1A5R", "1A66", "1A03",
    "1AHD", "1AUD", "1CFF", "2LVP", "2LVQ", "2MT6", "5IAY", "6BHN", "6BHO", "2M7U",
    "2M7V", "2K0Q", "2KM0", "2LEL", "2KOO", "2KOP", "2KOR", "2KOS", "2KSC", "4L2M",
    "4MAX", "2K5U", "2KSQ", "2KT2", "2KT3", "1AO8", "1BZF", "1DIS", "1LUD", "2HQP",
    "2L28", "2KYC", "2KYF", "2L61", "2L62", "1AP4", "1LXF", "1MXL", "2JXL", "2KFX",
    "1IH0", "1OZS", "2KDH", "1F54", "1F55", "2LIR", "2LIT", "2LKC", "2LKD", "2L67",
    "2LKK", "3B2H", "3VG3", "3VG5", "1BA9", "1KMG", "2M0Z", "2M10", "1AKD", "1CP4",
    "1GEM", "1DZ4", "1DZ6", "1DZ8", "5B03", "5B0L", "5GWW", "5TUF", "5TUI", "5IQT",
    "5TRQ", "5IQV", "7D09", "7D0A", "4LVC", "5M67", "5UPQ", "5UPS", "4XCK", "4XDA",
    "6YA3", "6YA4", "6Y9U", "6YAG", "5UZC", "5UZS", "5JGK", "5JGL", "3PJG", "3PLR",
    "4X7M", "4X7R", "4YHA", "5TT8", "5TT3"
]

OUT_DIR     = pathlib.Path("training/pdbs")  # change to your preferred folder
FILE_FORMAT = "pdb"                         # "pdb", "pdb.gz", "cif", or "cif.gz"
OVERWRITE   = False                         # True = re-download even if file exists

# ─── Implementation ─────────────────────────────────────────────────────────────

BASE_URL = "https://files.rcsb.org/download/{}"   # RCSB file template


def download_one(pdb_id: str, verbose=True) -> None:
    pdb_id = pdb_id.lower().strip()
    url  = BASE_URL.format(f"{pdb_id}.{FILE_FORMAT}")
    dest = OUT_DIR / f"{pdb_id}.{FILE_FORMAT}"

    if dest.exists() and not OVERWRITE:
        print(f"✓ {dest} already exists; skipping.") if verbose else None
        return

    print(f"↓ {pdb_id}  →  {dest}") if verbose else None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"✗ failed to download {pdb_id}: {exc}", file=sys.stderr)
        return False

    dest.write_bytes(r.content)
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdb_id in PDB_IDS:
        download_one(pdb_id)


if __name__ == "__main__":
    main()
