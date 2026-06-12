from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from .discovery import Entry


@dataclass(frozen=True)
class PreparedEntry:
    pdb_id: str
    source_path: Path
    structure_path: Path
    sequence: str
    sequence_length: int
    chain_id: str
    status: str = "ok"
    error: str = ""


@dataclass(frozen=True)
class PreparedInputs:
    entries: list[PreparedEntry]
    fasta_path: Path
    structures_dir: Path
    manifest_path: Path


def extract_representative_sequence(path: Path) -> tuple[str, str]:
    """Extract the longest protein-chain sequence using Biotite.

    Foldseek consumes the native structure file directly. This parser is only for
    producing the FASTA required by MMseqs.
    """
    try:
        import biotite.structure as struc
        from biotite.structure.io import load_structure
    except ImportError as exc:
        raise RuntimeError(
            "Biotite is required for sequence extraction. Run `uv sync` first."
        ) from exc

    atoms = load_structure(str(path))
    if atoms.__class__.__name__ == "AtomArrayStack":
        atoms = atoms[0]

    atoms = atoms[struc.filter_amino_acids(atoms)]
    if len(atoms) == 0:
        raise ValueError("no amino-acid atoms found")

    sequences, chain_starts = struc.to_sequence(atoms, allow_hetero=True)
    candidates: list[tuple[int, tuple[int, ...], str, str]] = []
    for seq, start in zip(sequences, chain_starts, strict=True):
        sequence = str(seq).replace("*", "X").upper()
        if not sequence:
            continue
        chain_id = str(atoms.chain_id[int(start)] or ".")
        candidates.append((len(sequence), _reverse_sort_key(chain_id), chain_id, sequence))

    if not candidates:
        raise ValueError("no protein sequence could be extracted")

    _, _, chain_id, sequence = max(candidates)
    return sequence, chain_id


def prepare_inputs(entries: list[Entry], out_dir: Path) -> PreparedInputs:
    work_dir = out_dir / "work"
    structures_dir = work_dir / "structures"
    fasta_path = work_dir / "sequences.fasta"
    manifest_path = out_dir / "manifest.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedEntry] = []
    manifest_rows: list[dict[str, str | int]] = []
    for entry in entries:
        try:
            sequence, chain_id = extract_representative_sequence(entry.path)
            structure_path = structures_dir / f"{entry.pdb_id}{entry.path.suffix.lower()}"
            _link_or_copy(entry.path, structure_path)
            prepared_entry = PreparedEntry(
                pdb_id=entry.pdb_id,
                source_path=entry.path,
                structure_path=structure_path,
                sequence=sequence,
                sequence_length=len(sequence),
                chain_id=chain_id,
            )
            prepared.append(prepared_entry)
            manifest_rows.append(_manifest_row(prepared_entry))
        except Exception as exc:
            manifest_rows.append(
                {
                    "pdb_id": entry.pdb_id,
                    "source_path": str(entry.path),
                    "structure_path": "",
                    "format": entry.format,
                    "sequence_length": 0,
                    "chain_id": "",
                    "status": "sequence_error",
                    "error": str(exc),
                }
            )

    _write_manifest(manifest_path, manifest_rows)
    if not prepared:
        raise RuntimeError(f"no usable protein entries found; see {manifest_path}")

    _write_fasta(fasta_path, prepared)
    return PreparedInputs(
        entries=prepared,
        fasta_path=fasta_path,
        structures_dir=structures_dir,
        manifest_path=manifest_path,
    )


def _manifest_row(entry: PreparedEntry) -> dict[str, str | int]:
    return {
        "pdb_id": entry.pdb_id,
        "source_path": str(entry.source_path),
        "structure_path": str(entry.structure_path),
        "format": entry.source_path.suffix.lstrip("."),
        "sequence_length": entry.sequence_length,
        "chain_id": entry.chain_id,
        "status": entry.status,
        "error": entry.error,
    }


def _write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "pdb_id",
        "source_path",
        "structure_path",
        "format",
        "sequence_length",
        "chain_id",
        "status",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_fasta(path: Path, entries: list[PreparedEntry]) -> None:
    with path.open("w") as handle:
        for entry in entries:
            handle.write(f">{entry.pdb_id}\n")
            for i in range(0, len(entry.sequence), 80):
                handle.write(f"{entry.sequence[i : i + 80]}\n")


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _reverse_sort_key(value: str) -> tuple[int, ...]:
    # Used with max(): shorter/lower lexical values win ties.
    return tuple(-ord(ch) for ch in value)
