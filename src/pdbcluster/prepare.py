from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from .discovery import Entry


@dataclass(frozen=True)
class PreparedChain:
    chain_uid: str
    pdb_id: str
    chain_id: str
    sequence: str
    sequence_length: int


@dataclass(frozen=True)
class PreparedEntry:
    pdb_id: str
    source_path: Path
    structure_path: Path
    chains: tuple[PreparedChain, ...]
    sequence_length: int
    status: str = "ok"
    error: str = ""


@dataclass(frozen=True)
class PreparedInputs:
    entries: list[PreparedEntry]
    chains: list[PreparedChain]
    fasta_path: Path
    structures_dir: Path
    manifest_path: Path


def extract_chain_sequences(path: Path) -> list[tuple[str, str]]:
    """Extract all protein-chain sequences from a PDB/mmCIF file."""
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
    chains: list[tuple[str, str]] = []
    seen: set[str] = set()
    for seq, start in zip(sequences, chain_starts, strict=True):
        sequence = str(seq).replace("*", "X").upper()
        if not sequence:
            continue
        chain_id = str(atoms.chain_id[int(start)] or ".")
        key = chain_id
        suffix = 1
        while key in seen:
            suffix += 1
            key = f"{chain_id}.{suffix}"
        seen.add(key)
        chains.append((key, sequence))

    if not chains:
        raise ValueError("no protein sequence could be extracted")
    return chains


def prepare_inputs(entries: list[Entry], out_dir: Path) -> PreparedInputs:
    work_dir = out_dir / "work"
    structures_dir = work_dir / "structures"
    fasta_path = work_dir / "chains.fasta"
    manifest_path = out_dir / "manifest.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedEntry] = []
    chains: list[PreparedChain] = []
    manifest_rows: list[dict[str, str | int]] = []
    for entry in entries:
        try:
            extracted = extract_chain_sequences(entry.path)
            entry_chains = tuple(
                PreparedChain(
                    chain_uid=f"{entry.pdb_id}__chain{index:04d}",
                    pdb_id=entry.pdb_id,
                    chain_id=chain_id,
                    sequence=sequence,
                    sequence_length=len(sequence),
                )
                for index, (chain_id, sequence) in enumerate(extracted, 1)
            )
            structure_path = structures_dir / f"{entry.pdb_id}{entry.path.suffix.lower()}"
            _link_or_copy(entry.path, structure_path)
            prepared_entry = PreparedEntry(
                pdb_id=entry.pdb_id,
                source_path=entry.path,
                structure_path=structure_path,
                chains=entry_chains,
                sequence_length=sum(chain.sequence_length for chain in entry_chains),
            )
            prepared.append(prepared_entry)
            chains.extend(entry_chains)
            manifest_rows.extend(_manifest_rows(prepared_entry))
        except Exception as exc:
            manifest_rows.append(
                {
                    "pdb_id": entry.pdb_id,
                    "chain_uid": "",
                    "chain_id": "",
                    "source_path": str(entry.path),
                    "structure_path": "",
                    "format": entry.format,
                    "sequence_length": 0,
                    "status": "sequence_error",
                    "error": str(exc),
                }
            )

    _write_manifest(manifest_path, manifest_rows)
    if not prepared:
        raise RuntimeError(f"no usable protein entries found; see {manifest_path}")

    _write_fasta(fasta_path, chains)
    return PreparedInputs(
        entries=prepared,
        chains=chains,
        fasta_path=fasta_path,
        structures_dir=structures_dir,
        manifest_path=manifest_path,
    )


def _manifest_rows(entry: PreparedEntry) -> list[dict[str, str | int]]:
    return [
        {
            "pdb_id": entry.pdb_id,
            "chain_uid": chain.chain_uid,
            "chain_id": chain.chain_id,
            "source_path": str(entry.source_path),
            "structure_path": str(entry.structure_path),
            "format": entry.source_path.suffix.lstrip("."),
            "sequence_length": chain.sequence_length,
            "status": entry.status,
            "error": entry.error,
        }
        for chain in entry.chains
    ]


def _write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "pdb_id",
        "chain_uid",
        "chain_id",
        "source_path",
        "structure_path",
        "format",
        "sequence_length",
        "status",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_fasta(path: Path, chains: list[PreparedChain]) -> None:
    with path.open("w") as handle:
        for chain in chains:
            handle.write(f">{chain.chain_uid} pdb_id={chain.pdb_id} chain_id={chain.chain_id}\n")
            for i in range(0, len(chain.sequence), 80):
                handle.write(f"{chain.sequence[i : i + 80]}\n")


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)
