from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = (".cif", ".pdb")


@dataclass(frozen=True)
class Entry:
    pdb_id: str
    path: Path
    format: str


def normalize_item_id(value: str) -> str:
    """Normalize a tool/file identifier to the pipeline item ID."""
    name = Path(value.strip().split()[0]).name
    lowered = name.lower()
    for suffix in (".cif.gz", ".pdb.gz", ".cif", ".pdb"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_entries(data_dir: Path) -> list[Entry]:
    """Find entries in <data_dir>/<pdb_id>/<pdb_id>_final.cif|pdb.

    CIF is preferred over PDB when both are present.
    """
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data directory is not a directory: {data_dir}")

    entries: list[Entry] = []
    seen: set[str] = set()
    for entry_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        pdb_id = entry_dir.name
        candidates = [
            entry_dir / f"{pdb_id}_final.cif",
            entry_dir / f"{pdb_id}_final.pdb",
            entry_dir / f"{pdb_id.lower()}_final.cif",
            entry_dir / f"{pdb_id.lower()}_final.pdb",
            entry_dir / f"{pdb_id.upper()}_final.cif",
            entry_dir / f"{pdb_id.upper()}_final.pdb",
        ]
        selected = next((path for path in candidates if path.exists()), None)
        if selected is None:
            continue

        item_id = normalize_item_id(pdb_id)
        if item_id in seen:
            raise ValueError(f"duplicate normalized PDB ID found: {item_id}")
        seen.add(item_id)
        entries.append(Entry(pdb_id=item_id, path=selected.resolve(), format=selected.suffix[1:]))

    return entries
