#!/usr/bin/env python3
"""Query RCSB PDB, then download matching PDB-REDO entries efficiently.

What this does
--------------
1. Runs an RCSB Search API query with configurable filters for:
   - experimental method
   - resolution
   - R-free
   - deposited polymer residue count
   - polymer entity type
2. Retrieves all matching PDB entry IDs.
3. Saves the IDs to <base_dir>/ids.txt.
4. Downloads one PDB-REDO ZIP per entry in parallel.
5. Extracts only the requested files into:
      <base_dir>/<pdb_id>/

Default extracted files per entry:
- <pdb_id>_final.cif
- <pdb_id>_final.mtz
- <pdb_id>_final.json
- data.json

Optional extracted file:
- <pdb_id>_final.pdb   (not available for all PDB-REDO entries)

Why this is faster than downloading individual files
----------------------------------------------------
- 1 HTTP request per entry instead of 4-5 requests per entry.
- Parallel downloads.
- Connection reuse with retries.
- Graceful handling of entries that do not have <pdb_id>_final.pdb.

Requirements
------------
- Python 3.9+
- requests

Example
-------
python download_pdb_redo.py /path/to/output --workers 16 --include-pdb --method "X-RAY DIFFRACTION" --max-resolution 3.0 --max-rfree 0.25 --min-residues 50 --max-residues 500 --polymer-entity-type "Protein (only)"
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_REDO_ZIP_URL = "https://pdb-redo.eu/db/{pdb_id}/zipped"

PRINT_LOCK = threading.Lock()
TLS = threading.local()


class DownloadError(RuntimeError):
    """Raised when an entry download or extraction fails."""


def eprint(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, file=sys.stderr, **kwargs)


def make_retrying_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        allowed_methods=frozenset({"GET", "POST"}),
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": "pdbredo-query-download/1.0 (+https://search.rcsb.org/; +https://pdb-redo.eu/)"
        }
    )
    return s


def get_session() -> requests.Session:
    session = getattr(TLS, "session", None)
    if session is None:
        session = make_retrying_session()
        TLS.session = session
    return session


def post_json(url: str, payload: dict, timeout: tuple[int, int] = (30, 300)) -> dict:
    session = get_session()
    resp = session.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        snippet = resp.text[:1000] if resp.text else ""
        raise RuntimeError(f"HTTP error from {url}: {exc}\nResponse snippet: {snippet}") from exc
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}") from exc


def extract_ids(result_set) -> List[str]:
    ids: List[str] = []
    for item in result_set or []:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            # Handle alternate response shapes defensively.
            if "identifier" in item:
                ids.append(str(item["identifier"]))
            elif "id" in item:
                ids.append(str(item["id"]))
            else:
                raise RuntimeError(f"Unrecognized result_set item: {item!r}")
        else:
            raise RuntimeError(f"Unrecognized result_set item type: {type(item)!r}")
    return ids


def build_query(
    *,
    method: str,
    max_resolution: float,
    max_rfree: float,
    min_residues: int,
    max_residues: int,
    polymer_entity_type: str,
) -> dict:
    return {
        "return_type": "entry",
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": method,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "refine.ls_R_factor_R_free",
                        "operator": "less",
                        "value": max_rfree,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "range",
                        "value": {
                            "from": min_residues,
                            "to": max_residues,
                            "include_lower": True,
                            "include_upper": True,
                        },
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": polymer_entity_type,
                    },
                },
            ],
        },
    }


def fetch_total_count(query: dict) -> int:
    payload = copy.deepcopy(query)
    payload["request_options"] = {"return_counts": True}
    data = post_json(RCSB_SEARCH_URL, payload)
    total = data.get("total_count")
    if not isinstance(total, int):
        raise RuntimeError(f"Could not read total_count from response: {data}")
    return total


def fetch_all_ids(query: dict, page_size: int = 10000) -> List[str]:
    total = fetch_total_count(query)
    ids: List[str] = []
    start = 0

    while start < total:
        payload = copy.deepcopy(query)
        payload["request_options"] = {
            "results_verbosity": "compact",
            "paginate": {"start": start, "rows": page_size},
        }
        data = post_json(RCSB_SEARCH_URL, payload)
        batch = extract_ids(data.get("result_set", []))

        if not batch:
            raise RuntimeError(
                f"Pagination stopped early at start={start}; expected {total} total hits."
            )

        ids.extend(batch)
        start += len(batch)

        with PRINT_LOCK:
            print(f"Fetched {len(ids)}/{total} IDs from RCSB", file=sys.stderr)

    # Deduplicate while preserving order, just in case.
    deduped = list(dict.fromkeys(x.lower() for x in ids))
    return deduped


def desired_files_for_entry(pdb_id: str, include_pdb: bool) -> set[str]:
    wanted = {
        f"{pdb_id}_final.cif",
        f"{pdb_id}_final.mtz",
        f"{pdb_id}_final.json",
        "data.json",
    }
    if include_pdb:
        wanted.add(f"{pdb_id}_final.pdb")
    return wanted


def core_files_for_entry(pdb_id: str) -> set[str]:
    return {
        f"{pdb_id}_final.cif",
        f"{pdb_id}_final.mtz",
        f"{pdb_id}_final.json",
        "data.json",
    }


def entry_is_complete(entry_dir: Path, pdb_id: str, include_pdb: bool) -> bool:
    core = core_files_for_entry(pdb_id)
    if not all((entry_dir / name).exists() for name in core):
        return False
    if include_pdb:
        # PDB format is optional in the PDB-REDO archive, so do not require it
        # to consider an entry complete.
        return True
    return True


def stream_to_tempfile(url: str) -> str:
    session = get_session()
    with session.get(url, stream=True, timeout=(30, 600)) as resp:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            snippet = resp.text[:1000] if resp.text else ""
            raise DownloadError(f"Failed to download {url}: {exc}\nResponse snippet: {snippet}") from exc

        with tempfile.NamedTemporaryFile(prefix="pdbredo_", suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
    return tmp_path


def atomic_copy_from_zip(zf: zipfile.ZipFile, member_name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with zf.open(member_name) as src:
            shutil.copyfileobj(src, tmp, length=1024 * 1024)
    tmp_path.replace(dest)


def _remove_entry_dir_if_empty(entry_dir: Path) -> None:
    """Remove the entry directory if it exists and contains no files."""
    try:
        if entry_dir.is_dir() and not any(entry_dir.iterdir()):
            entry_dir.rmdir()
    except OSError:
        pass


def download_one_entry(pdb_id: str, base_dir: Path, include_pdb: bool, force: bool) -> dict:
    pdb_id = pdb_id.lower()
    entry_dir = base_dir / pdb_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    wanted = desired_files_for_entry(pdb_id, include_pdb)
    core = core_files_for_entry(pdb_id)

    if not force and entry_is_complete(entry_dir, pdb_id, include_pdb):
        return {
            "pdb_id": pdb_id,
            "status": "skipped",
            "missing_optional": [name for name in wanted - core if not (entry_dir / name).exists()],
        }

    zip_url = PDB_REDO_ZIP_URL.format(pdb_id=pdb_id)
    try:
        tmp_zip = stream_to_tempfile(zip_url)
    except DownloadError:
        # Clean up the empty directory before re-raising.
        _remove_entry_dir_if_empty(entry_dir)
        raise

    found: set[str] = set()
    try:
        with zipfile.ZipFile(tmp_zip) as zf:
            basename_to_member = {Path(name).name: name for name in zf.namelist()}

            for short_name in wanted:
                member = basename_to_member.get(short_name)
                if member is None:
                    continue
                atomic_copy_from_zip(zf, member, entry_dir / short_name)
                found.add(short_name)
    except zipfile.BadZipFile as exc:
        _remove_entry_dir_if_empty(entry_dir)
        raise DownloadError(f"Corrupt ZIP received for {pdb_id}") from exc
    finally:
        try:
            os.remove(tmp_zip)
        except OSError:
            pass

    missing_core = sorted(core - found)
    missing_optional = sorted((wanted - core) - found)

    if missing_core:
        _remove_entry_dir_if_empty(entry_dir)
        raise DownloadError(f"Missing required files for {pdb_id}: {', '.join(missing_core)}")

    return {
        "pdb_id": pdb_id,
        "status": "downloaded",
        "missing_optional": missing_optional,
    }


def write_ids(ids: Iterable[str], path: Path) -> None:
    text = "\n".join(x.lower() for x in ids) + "\n"
    path.write_text(text)


def write_manifest(results: list[dict], path: Path) -> None:
    lines = ["pdb_id\tstatus\tmissing_optional\terror"]
    for r in results:
        lines.append(
            "\t".join(
                [
                    r.get("pdb_id", ""),
                    r.get("status", ""),
                    ",".join(r.get("missing_optional", [])),
                    r.get("error", ""),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an RCSB query and download matching PDB-REDO entries efficiently.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "base_dir",
        type=Path,
        help="Base output directory. Files are saved under <base_dir>/<pdb_id>/",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4) * 2),
        help="Number of concurrent PDB-REDO downloads (default: %(default)s)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=10000,
        help="RCSB pagination size (max 10000; default: %(default)s)",
    )
    p.add_argument(
        "--include-pdb",
        action="store_true",
        help="Also try to extract <pdb_id>_final.pdb when present. Missing PDB files are not treated as failures.",
    )
    p.add_argument(
        "--method",
        default="X-RAY DIFFRACTION",
        help="RCSB experimental method filter.",
    )
    p.add_argument(
        "--max-resolution",
        type=float,
        default=3.0,
        help="Maximum combined resolution in Angstrom.",
    )
    p.add_argument(
        "--max-rfree",
        type=float,
        default=0.25,
        help="Maximum refinement R-free value.",
    )
    p.add_argument(
        "--min-residues",
        type=int,
        default=50,
        help="Minimum deposited polymer monomer count.",
    )
    p.add_argument(
        "--max-residues",
        type=int,
        default=500,
        help="Maximum deposited polymer monomer count.",
    )
    p.add_argument(
        "--polymer-entity-type",
        default="Protein (only)",
        help="Exact-match polymer entity type filter.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload entries even if required output files already exist.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failed entry download.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir: Path = args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.page_size < 1 or args.page_size > 10000:
        raise SystemExit("--page-size must be between 1 and 10000")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.min_residues > args.max_residues:
        raise SystemExit("--min-residues must be <= --max-residues")

    t0 = time.time()
    print("Querying RCSB for matching entry IDs...", file=sys.stderr)
    query = build_query(
        method=args.method,
        max_resolution=args.max_resolution,
        max_rfree=args.max_rfree,
        min_residues=args.min_residues,
        max_residues=args.max_residues,
        polymer_entity_type=args.polymer_entity_type,
    )
    ids = fetch_all_ids(query, page_size=args.page_size)
    print(f"Found {len(ids)} matching entries.", file=sys.stderr)

    ids_path = base_dir / "ids.txt"
    write_ids(ids, ids_path)
    print(f"Wrote {ids_path}", file=sys.stderr)

    results: list[dict] = []
    failures = 0
    downloaded = 0
    skipped = 0

    print(
        f"Downloading PDB-REDO ZIPs with {args.workers} worker(s)...",
        file=sys.stderr,
    )

    dl_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one_entry, pdb_id, base_dir, args.include_pdb, args.force): pdb_id
            for pdb_id in ids
        }

        for i, fut in enumerate(as_completed(futures), start=1):
            pdb_id = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - want broad capture for manifest
                failures += 1
                result = {
                    "pdb_id": pdb_id,
                    "status": "failed",
                    "missing_optional": [],
                    "error": str(exc),
                }
                eprint(f"[{i}/{len(ids)}] FAILED {pdb_id}: {exc}")
                results.append(result)
                if args.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    break
                continue

            results.append(result)
            if result["status"] == "downloaded":
                downloaded += 1
            elif result["status"] == "skipped":
                skipped += 1

            elapsed_dl = time.time() - dl_start
            rate = i / elapsed_dl if elapsed_dl > 0 else 0
            remaining = len(ids) - i
            eta = remaining / rate if rate > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

            suffix = ""
            if result.get("missing_optional"):
                suffix = f" (missing optional: {', '.join(result['missing_optional'])})"
            print(
                f"[{i}/{len(ids)}] {result['status'].upper()} {pdb_id}{suffix}"
                f"  ({rate:.1f} entries/s, ETA {eta_str})",
                file=sys.stderr,
            )

    manifest_path = base_dir / "download_manifest.tsv"
    write_manifest(results, manifest_path)

    elapsed = time.time() - t0
    print("", file=sys.stderr)
    print(f"Done in {elapsed:.1f} s", file=sys.stderr)
    print(f"Total IDs      : {len(ids)}", file=sys.stderr)
    print(f"Downloaded     : {downloaded}", file=sys.stderr)
    print(f"Skipped        : {skipped}", file=sys.stderr)
    print(f"Failed         : {failures}", file=sys.stderr)
    print(f"IDs file       : {ids_path}", file=sys.stderr)
    print(f"Manifest       : {manifest_path}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
