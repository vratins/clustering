#!/usr/bin/env python3
"""Download PDB-REDO files for a list of PDB IDs.

Input file format: one entry per line, e.g. '1abc_final'
Downloads per entry: <pdb_id>_final.pdb, <pdb_id>_final.cif,
                     <pdb_id>_final.mtz, <pdb_id>_final.json, data.json
Output structure: <base_dir>/<pdb_id>/
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://pdb-redo.eu/db"


def download_file(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  URL error ({e.reason}): {url}", file=sys.stderr)
        return False


def download_entry(pdb_id: str, base_dir: Path) -> None:
    out_dir = base_dir / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f"{pdb_id}_final.pdb",
        f"{pdb_id}_final.cif",
        f"{pdb_id}_final.mtz",
        f"{pdb_id}_final.json",
        "data.json",
    ]

    for filename in files:
        url = f"{BASE_URL}/{pdb_id}/{filename}"
        dest = out_dir / filename
        if dest.exists():
            print(f"  [skip] {filename} already exists")
            continue
        ok = download_file(url, dest)
        status = "ok" if ok else "FAILED"
        print(f"  [{status}] {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PDB-REDO files for a list of PDB IDs.")
    parser.add_argument("input_file", help="Text file with one '<pdb_id>_final' entry per line")
    parser.add_argument("base_dir", help="Base output directory; files go to <base_dir>/<pdb_id>/")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    base_dir = Path(args.base_dir)

    entries = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]

    for entry in tqdm(entries):
        # Accept either 'abcd_final' or bare 'abcd'
        pdb_id = entry.removesuffix("_final").lower()
        print(f"Downloading {pdb_id} ...")
        download_entry(pdb_id, base_dir)

    print("Done.")


if __name__ == "__main__":
    main()

