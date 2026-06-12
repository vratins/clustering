from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .discovery import normalize_item_id
from .prepare import PreparedEntry


@dataclass(frozen=True)
class FusionThresholds:
    seq_id: float
    seq_cov: float
    tm: float
    struct_cov: float


SEQ_FIELDS = ["query", "target", "fident", "qcov", "tcov", "alnlen", "evalue", "bits"]
STRUCT_FIELDS = [
    "query",
    "target",
    "alntmscore",
    "qtmscore",
    "ttmscore",
    "lddt",
    "prob",
    "fident",
    "qcov",
    "tcov",
    "alnlen",
    "evalue",
    "bits",
]


def load_cluster_assignments(path: Path, known_ids: set[str]) -> dict[str, str]:
    assignments = {item_id: item_id for item_id in known_ids}
    if not path.exists():
        return assignments

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep = _known_id(parts[0], known_ids)
            member = _known_id(parts[1], known_ids)
            if rep is None or member is None:
                continue
            assignments[member] = rep
    return assignments


def fuse_edges(
    entries: list[PreparedEntry],
    seq_edges_path: Path,
    struct_edges_path: Path,
    seq_clusters_path: Path,
    struct_clusters_path: Path,
    out_dir: Path,
    thresholds: FusionThresholds,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    known_ids = {entry.pdb_id for entry in entries}
    lengths = {entry.pdb_id: entry.sequence_length for entry in entries}
    seq_clusters = load_cluster_assignments(seq_clusters_path, known_ids)
    struct_clusters = load_cluster_assignments(struct_clusters_path, known_ids)

    seq_edges = _load_sequence_edges(seq_edges_path, known_ids)
    struct_edges = _load_structure_edges(struct_edges_path, known_ids)
    passing: list[dict[str, str | float]] = []

    parent = {item_id: item_id for item_id in known_ids}
    for pair, seq in seq_edges.items():
        struct = struct_edges.get(pair)
        if struct is None:
            continue
        if not _passes(seq, struct, thresholds):
            continue
        a, b = pair
        union(parent, a, b)
        passing.append(
            {
                "item_a": a,
                "item_b": b,
                "seq_identity": seq["fident"],
                "seq_qcov": seq["qcov"],
                "seq_tcov": seq["tcov"],
                "qtmscore": struct["qtmscore"],
                "ttmscore": struct["ttmscore"],
                "struct_qcov": struct["qcov"],
                "struct_tcov": struct["tcov"],
                "alntmscore": struct["alntmscore"],
                "lddt": struct["lddt"],
                "prob": struct["prob"],
            }
        )

    _write_final_edges(out_dir / "final_edges.tsv", passing)
    _write_final_clusters(
        out_dir / "final_clusters.tsv",
        known_ids,
        parent,
        lengths,
        passing,
        seq_clusters,
        struct_clusters,
    )


def find_cluster_tsv(prefix: Path) -> Path:
    candidates = [
        prefix.with_name(prefix.name + "_cluster.tsv"),
        prefix.with_name(prefix.name + "_clu.tsv"),
    ]
    candidates.extend(sorted(prefix.parent.glob(prefix.name + "*_cluster.tsv")))
    candidates.extend(sorted(prefix.parent.glob(prefix.name + "*_clu.tsv")))
    candidates.extend(sorted(prefix.parent.glob(prefix.name + "*.tsv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def union(parent: dict[str, str], a: str, b: str) -> None:
    root_a = find(parent, a)
    root_b = find(parent, b)
    if root_a == root_b:
        return
    if root_b < root_a:
        root_a, root_b = root_b, root_a
    parent[root_b] = root_a


def find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _passes(
    seq: dict[str, float],
    struct: dict[str, float],
    thresholds: FusionThresholds,
) -> bool:
    return (
        seq["fident"] >= thresholds.seq_id
        and seq["qcov"] >= thresholds.seq_cov
        and seq["tcov"] >= thresholds.seq_cov
        and min(struct["qtmscore"], struct["ttmscore"]) >= thresholds.tm
        and struct["qcov"] >= thresholds.struct_cov
        and struct["tcov"] >= thresholds.struct_cov
    )


def _load_sequence_edges(
    path: Path, known_ids: set[str]
) -> dict[tuple[str, str], dict[str, float]]:
    rows = _read_tool_tsv(path, SEQ_FIELDS)
    edges: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        query = _known_id(row["query"], known_ids)
        target = _known_id(row["target"], known_ids)
        if query is None or target is None or query == target:
            continue
        pair = tuple(sorted((query, target)))
        metrics = {
            "fident": _fraction(row["fident"]),
            "qcov": _fraction(row["qcov"]),
            "tcov": _fraction(row["tcov"]),
        }
        if pair not in edges or metrics["fident"] > edges[pair]["fident"]:
            edges[pair] = metrics
    return edges


def _load_structure_edges(
    path: Path, known_ids: set[str]
) -> dict[tuple[str, str], dict[str, float]]:
    rows = _read_tool_tsv(path, STRUCT_FIELDS)
    edges: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        query = _known_id(row["query"], known_ids)
        target = _known_id(row["target"], known_ids)
        if query is None or target is None or query == target:
            continue
        pair = tuple(sorted((query, target)))
        metrics = {
            "alntmscore": _fraction(row["alntmscore"]),
            "qtmscore": _fraction(row["qtmscore"]),
            "ttmscore": _fraction(row["ttmscore"]),
            "lddt": _fraction(row["lddt"]),
            "prob": _fraction(row["prob"]),
            "qcov": _fraction(row["qcov"]),
            "tcov": _fraction(row["tcov"]),
        }
        score = min(metrics["qtmscore"], metrics["ttmscore"])
        previous_score = (
            min(edges[pair]["qtmscore"], edges[pair]["ttmscore"]) if pair in edges else -1.0
        )
        if score > previous_score:
            edges[pair] = metrics
    return edges


def _read_tool_tsv(path: Path, default_fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row]
    if not rows:
        return []
    first = rows[0]
    if {"query", "target"}.issubset(first):
        return [dict(zip(first, row, strict=False)) for row in rows[1:]]
    return [dict(zip(default_fields, row, strict=False)) for row in rows]


def _known_id(value: str, known_ids: set[str]) -> str | None:
    normalized = normalize_item_id(value)
    if normalized in known_ids:
        return normalized
    lowered = normalized.lower()
    return lowered if lowered in known_ids else None


def _fraction(value: str) -> float:
    number = float(value)
    if number > 1.0 and number <= 100.0:
        return number / 100.0
    return number


def _write_final_edges(path: Path, rows: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "item_a",
        "item_b",
        "seq_identity",
        "seq_qcov",
        "seq_tcov",
        "qtmscore",
        "ttmscore",
        "struct_qcov",
        "struct_tcov",
        "alntmscore",
        "lddt",
        "prob",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["item_a"]), str(row["item_b"]))))


def _write_final_clusters(
    path: Path,
    known_ids: set[str],
    parent: dict[str, str],
    lengths: dict[str, int],
    passing_edges: list[dict[str, str | float]],
    seq_clusters: dict[str, str],
    struct_clusters: dict[str, str],
) -> None:
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for item_id in sorted(known_ids):
        members_by_root[find(parent, item_id)].append(item_id)

    degree = {item_id: 0 for item_id in known_ids}
    for edge in passing_edges:
        degree[str(edge["item_a"])] += 1
        degree[str(edge["item_b"])] += 1

    cluster_ids: dict[str, str] = {}
    for index, root in enumerate(
        sorted(members_by_root, key=lambda key: min(members_by_root[key])), 1
    ):
        cluster_ids[root] = f"C{index:06d}"

    with path.open("w", newline="") as handle:
        fieldnames = [
            "pdb_id",
            "final_cluster",
            "final_representative",
            "sequence_cluster",
            "structure_cluster",
            "sequence_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for root, members in sorted(members_by_root.items(), key=lambda item: min(item[1])):
            representative = max(
                members, key=lambda item: (degree[item], lengths[item], _reverse(item))
            )
            for item_id in sorted(members):
                writer.writerow(
                    {
                        "pdb_id": item_id,
                        "final_cluster": cluster_ids[root],
                        "final_representative": representative,
                        "sequence_cluster": seq_clusters.get(item_id, item_id),
                        "structure_cluster": struct_clusters.get(item_id, item_id),
                        "sequence_length": lengths[item_id],
                    }
                )


def _reverse(value: str) -> tuple[int, ...]:
    return tuple(-ord(ch) for ch in value)
