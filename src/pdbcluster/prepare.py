from __future__ import annotations

import csv
import json
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
    cached: bool = False


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
    params_path = work_dir / "prepare.params.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    cache_params = _prepare_cache_params(entries)
    if _prepare_cache_valid(params_path, manifest_path, fasta_path, structures_dir, cache_params):
        return _load_prepared_inputs(manifest_path, fasta_path, structures_dir, cached=True)
    if _prepared_outputs_match_entries(manifest_path, fasta_path, structures_dir, entries):
        params_path.write_text(json.dumps(cache_params, indent=2, sort_keys=True) + "\n")
        return _load_prepared_inputs(manifest_path, fasta_path, structures_dir, cached=True)

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
    params_path.write_text(json.dumps(cache_params, indent=2, sort_keys=True) + "\n")
    return PreparedInputs(
        entries=prepared,
        chains=chains,
        fasta_path=fasta_path,
        structures_dir=structures_dir,
        manifest_path=manifest_path,
    )



def _prepare_cache_params(entries: list[Entry]) -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            {
                "pdb_id": entry.pdb_id,
                "path": str(entry.path),
                "format": entry.format,
                "size": entry.path.stat().st_size,
                "mtime_ns": entry.path.stat().st_mtime_ns,
            }
            for entry in entries
        ],
    }


def _prepare_cache_valid(
    params_path: Path,
    manifest_path: Path,
    fasta_path: Path,
    structures_dir: Path,
    params: dict[str, object],
) -> bool:
    if not params_path.exists() or not manifest_path.exists() or not fasta_path.exists():
        return False
    if not structures_dir.exists():
        return False
    try:
        cached = json.loads(params_path.read_text())
    except json.JSONDecodeError:
        return False
    if cached != params:
        return False

    try:
        prepared = _load_prepared_inputs(manifest_path, fasta_path, structures_dir)
    except (OSError, ValueError):
        return False
    return all(entry.structure_path.exists() for entry in prepared.entries)



def _prepared_outputs_match_entries(
    manifest_path: Path,
    fasta_path: Path,
    structures_dir: Path,
    entries: list[Entry],
) -> bool:
    if not manifest_path.exists() or not fasta_path.exists() or not structures_dir.exists():
        return False
    try:
        prepared = _load_prepared_inputs(manifest_path, fasta_path, structures_dir)
        rows = _read_manifest_rows(manifest_path)
    except (OSError, ValueError):
        return False

    rows_by_pdb: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_pdb.setdefault(row.get("pdb_id", ""), []).append(row)

    if {entry.pdb_id for entry in entries} != set(rows_by_pdb):
        return False
    for entry in entries:
        entry_rows = rows_by_pdb.get(entry.pdb_id, [])
        if not entry_rows:
            return False
        if any(row.get("source_path") != str(entry.path) for row in entry_rows):
            return False
        if any(row.get("format") != entry.format for row in entry_rows):
            return False

    newest_source_mtime = max((entry.path.stat().st_mtime_ns for entry in entries), default=0)
    oldest_output_mtime = min(manifest_path.stat().st_mtime_ns, fasta_path.stat().st_mtime_ns)
    if oldest_output_mtime < newest_source_mtime:
        return False
    return all(entry.structure_path.exists() for entry in prepared.entries)



def _load_prepared_inputs(
    manifest_path: Path,
    fasta_path: Path,
    structures_dir: Path,
    cached: bool = False,
) -> PreparedInputs:
    sequences = _read_fasta(fasta_path)
    rows_by_pdb: dict[str, list[dict[str, str]]] = {}
    for row in _read_manifest_rows(manifest_path):
        if row.get("status") != "ok" or not row.get("chain_uid"):
            continue
        rows_by_pdb.setdefault(row["pdb_id"], []).append(row)

    entries: list[PreparedEntry] = []
    chains: list[PreparedChain] = []
    for pdb_id in sorted(rows_by_pdb):
        rows = rows_by_pdb[pdb_id]
        entry_chains: list[PreparedChain] = []
        for row in rows:
            chain_uid = row["chain_uid"]
            sequence = sequences.get(chain_uid)
            if sequence is None:
                raise ValueError(f"missing FASTA sequence for {chain_uid}")
            chain = PreparedChain(
                chain_uid=chain_uid,
                pdb_id=pdb_id,
                chain_id=row["chain_id"],
                sequence=sequence,
                sequence_length=len(sequence),
            )
            entry_chains.append(chain)
            chains.append(chain)

        first = rows[0]
        entries.append(
            PreparedEntry(
                pdb_id=pdb_id,
                source_path=Path(first["source_path"]),
                structure_path=Path(first["structure_path"]),
                chains=tuple(entry_chains),
                sequence_length=sum(chain.sequence_length for chain in entry_chains),
            )
        )

    if not entries:
        raise ValueError(f"no cached protein entries found in {manifest_path}")
    return PreparedInputs(entries, chains, fasta_path, structures_dir, manifest_path, cached=cached)



def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            elif current_id is not None:
                chunks.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(chunks)
    return sequences

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
