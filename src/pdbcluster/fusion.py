from __future__ import annotations

import csv
import warnings
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
SEQ_EDGE_FIELDS = [
    "item_a",
    "item_b",
    "seq_identity",
    "seq_qcov",
    "seq_tcov",
    "seq_query_chain",
    "seq_target_chain",
    "seq_evidence_count",
]
STRUCT_EDGE_FIELDS = [
    "item_a",
    "item_b",
    "complex_qtm",
    "complex_ttm",
    "complex_qcov",
    "complex_tcov",
    "interface_lddt",
    "source_component",
]
REPORT_FIELDS = [
    "query",
    "target",
    "complex_qcov",
    "complex_tcov",
    "complex_qtm",
    "complex_ttm",
    "interface_lddt",
    "u",
    "t",
]


def collapse_sequence_edges(
    path: Path,
    chain_to_structure: dict[str, str],
    thresholds: FusionThresholds,
) -> dict[tuple[str, str], dict[str, str | float | int]]:
    edges: dict[tuple[str, str], dict[str, str | float | int]] = {}
    unmatched: set[str] = set()
    for row in _read_tool_tsv(path, SEQ_FIELDS):
        query_chain = row.get("query", "")
        target_chain = row.get("target", "")
        query = chain_to_structure.get(query_chain)
        target = chain_to_structure.get(target_chain)
        if query is None:
            unmatched.add(query_chain)
        if target is None:
            unmatched.add(target_chain)
        if query is None or target is None or query == target:
            continue

        metrics = {
            "seq_identity": _fraction(row.get("fident", "0")),
            "seq_qcov": _fraction(row.get("qcov", "0")),
            "seq_tcov": _fraction(row.get("tcov", "0")),
        }
        if (
            metrics["seq_identity"] < thresholds.seq_id
            or metrics["seq_qcov"] < thresholds.seq_cov
            or metrics["seq_tcov"] < thresholds.seq_cov
        ):
            continue

        item_a, item_b = sorted((query, target))
        if query == item_a:
            chain_a, chain_b = query_chain, target_chain
            qcov, tcov = metrics["seq_qcov"], metrics["seq_tcov"]
        else:
            chain_a, chain_b = target_chain, query_chain
            qcov, tcov = metrics["seq_tcov"], metrics["seq_qcov"]
        pair = (item_a, item_b)
        previous = edges.get(pair)
        evidence_count = int(previous["seq_evidence_count"]) + 1 if previous else 1
        candidate = {
            "item_a": item_a,
            "item_b": item_b,
            "seq_identity": metrics["seq_identity"],
            "seq_qcov": qcov,
            "seq_tcov": tcov,
            "seq_query_chain": chain_a,
            "seq_target_chain": chain_b,
            "seq_evidence_count": evidence_count,
        }
        if previous is None or float(candidate["seq_identity"]) > float(previous["seq_identity"]):
            edges[pair] = candidate
        else:
            previous["seq_evidence_count"] = evidence_count

    _warn_unmatched("MMseqs chain", unmatched)
    return edges


def write_sequence_edges(
    path: Path, edges: dict[tuple[str, str], dict[str, str | float | int]]
) -> None:
    _write_rows(path, SEQ_EDGE_FIELDS, edges.values())


def load_sequence_edges(path: Path) -> dict[tuple[str, str], dict[str, str | float | int]]:
    return _load_edge_table(path, SEQ_EDGE_FIELDS, int_fields={"seq_evidence_count"})


def sequence_components(
    known_ids: set[str], edges: dict[tuple[str, str], dict[str, str | float | int]]
) -> dict[str, str]:
    parent = {item_id: item_id for item_id in known_ids}
    for item_a, item_b in edges:
        union(parent, item_a, item_b)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for item_id in sorted(known_ids):
        members_by_root[find(parent, item_id)].append(item_id)

    assignments: dict[str, str] = {}
    for index, root in enumerate(
        sorted(members_by_root, key=lambda key: min(members_by_root[key])), 1
    ):
        component_id = f"S{index:06d}"
        for item_id in members_by_root[root]:
            assignments[item_id] = component_id
    return assignments


def parse_multimer_cluster_report(
    path: Path, known_ids: set[str], component_id: str
) -> dict[tuple[str, str], dict[str, str | float]]:
    edges: dict[tuple[str, str], dict[str, str | float]] = {}
    unmatched: set[str] = set()
    for row in _read_tool_tsv(path, REPORT_FIELDS):
        query = _known_id(row.get("query", ""), known_ids)
        target = _known_id(row.get("target", ""), known_ids)
        if query is None:
            unmatched.add(row.get("query", ""))
        if target is None:
            unmatched.add(row.get("target", ""))
        if query is None or target is None or query == target:
            continue

        pair = tuple(sorted((query, target)))
        metrics = {
            "item_a": pair[0],
            "item_b": pair[1],
            "complex_qtm": _fraction(row.get("complex_qtm", "0")),
            "complex_ttm": _fraction(row.get("complex_ttm", "0")),
            "complex_qcov": _fraction(row.get("complex_qcov", "0")),
            "complex_tcov": _fraction(row.get("complex_tcov", "0")),
            "interface_lddt": _fraction(row.get("interface_lddt", "0")),
            "source_component": component_id,
        }
        score = min(float(metrics["complex_qtm"]), float(metrics["complex_ttm"]))
        previous = edges.get(pair)
        previous_score = (
            min(float(previous["complex_qtm"]), float(previous["complex_ttm"]))
            if previous
            else -1.0
        )
        if score > previous_score:
            edges[pair] = metrics

    _warn_unmatched("Foldseek structure", unmatched)
    return edges


def write_structure_edges(path: Path, edges: dict[tuple[str, str], dict[str, str | float]]) -> None:
    _write_rows(path, STRUCT_EDGE_FIELDS, edges.values())


def load_structure_edges(path: Path) -> dict[tuple[str, str], dict[str, str | float]]:
    return _load_edge_table(path, STRUCT_EDGE_FIELDS)


def load_cluster_assignments(paths: list[Path], known_ids: set[str]) -> dict[str, str]:
    assignments = {item_id: item_id for item_id in known_ids}
    unmatched: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
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
                if rep is None:
                    unmatched.add(parts[0])
                if member is None:
                    unmatched.add(parts[1])
                if rep is not None and member is not None:
                    assignments[member] = rep
    _warn_unmatched("Foldseek cluster", unmatched)
    return assignments


def fuse_edges(
    entries: list[PreparedEntry],
    seq_edges_path: Path,
    struct_edges_path: Path,
    out_dir: Path,
    thresholds: FusionThresholds,
    sequence_component_by_item: dict[str, str] | None = None,
    structure_cluster_by_item: dict[str, str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    known_ids = {entry.pdb_id for entry in entries}
    lengths = {entry.pdb_id: entry.sequence_length for entry in entries}
    seq_edges = load_sequence_edges(seq_edges_path)
    struct_edges = load_structure_edges(struct_edges_path)
    sequence_component_by_item = sequence_component_by_item or {
        item_id: item_id for item_id in known_ids
    }
    structure_cluster_by_item = structure_cluster_by_item or {
        item_id: item_id for item_id in known_ids
    }

    passing: list[dict[str, str | float | int]] = []
    parent = {item_id: item_id for item_id in known_ids}
    for pair, seq in seq_edges.items():
        struct = struct_edges.get(pair)
        if struct is None or not _passes(seq, struct, thresholds):
            continue
        item_a, item_b = pair
        union(parent, item_a, item_b)
        passing.append({**seq, **struct})

    _write_final_edges(out_dir / "final_edges.tsv", passing)
    _write_final_clusters(
        out_dir / "final_clusters.tsv",
        known_ids,
        parent,
        lengths,
        passing,
        sequence_component_by_item,
        structure_cluster_by_item,
    )


def find_cluster_tsv(prefix: Path) -> Path:
    candidates = [
        prefix.with_name(prefix.name + "_cluster.tsv"),
        prefix.with_name(prefix.name + "_clu.tsv"),
    ]
    candidates.extend(sorted(prefix.parent.glob(prefix.name + "*_cluster.tsv")))
    candidates.extend(sorted(prefix.parent.glob(prefix.name + "*_clu.tsv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_cluster_report(prefix: Path) -> Path:
    candidates = [prefix.with_name(prefix.name + "_cluster_report")]
    candidates.extend(sorted(prefix.parent.glob(prefix.name + "*_cluster_report*")))
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
    seq: dict[str, str | float | int],
    struct: dict[str, str | float],
    thresholds: FusionThresholds,
) -> bool:
    return (
        float(seq["seq_identity"]) >= thresholds.seq_id
        and float(seq["seq_qcov"]) >= thresholds.seq_cov
        and float(seq["seq_tcov"]) >= thresholds.seq_cov
        and min(float(struct["complex_qtm"]), float(struct["complex_ttm"])) >= thresholds.tm
        and float(struct["complex_qcov"]) >= thresholds.struct_cov
        and float(struct["complex_tcov"]) >= thresholds.struct_cov
    )


def _read_tool_tsv(path: Path, default_fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row]
    if not rows:
        return []
    first = rows[0]
    if set(default_fields[:2]).issubset(first):
        return [dict(zip(first, row, strict=False)) for row in rows[1:]]
    return [dict(zip(default_fields, row, strict=False)) for row in rows]


def _load_edge_table(
    path: Path, fieldnames: list[str], int_fields: set[str] | None = None
) -> dict[tuple[str, str], dict[str, str | float | int]]:
    int_fields = int_fields or set()
    edges: dict[tuple[str, str], dict[str, str | float | int]] = {}
    for row in _read_tool_tsv(path, fieldnames):
        item_a = row.get("item_a", "")
        item_b = row.get("item_b", "")
        if not item_a or not item_b or item_a == item_b:
            continue
        parsed: dict[str, str | float | int] = {"item_a": item_a, "item_b": item_b}
        for key, value in row.items():
            if key in {"item_a", "item_b"}:
                continue
            if key in int_fields:
                parsed[key] = int(value)
            elif key.endswith("chain") or key == "source_component":
                parsed[key] = value
            else:
                parsed[key] = _fraction(value)
        edges[tuple(sorted((item_a, item_b)))] = parsed
    return edges


def _known_id(value: str, known_ids: set[str]) -> str | None:
    normalized = normalize_item_id(value)
    if normalized in known_ids:
        return normalized
    lowered = normalized.lower()
    return lowered if lowered in known_ids else None


def _fraction(value: str) -> float:
    if value == "":
        return 0.0
    number = float(value)
    if 1.0 < number <= 100.0:
        return number / 100.0
    return number


def _write_rows(
    path: Path, fieldnames: list[str], rows: object
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (str(row["item_a"]), str(row["item_b"])),
            )
        )


def _write_final_edges(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    _write_rows(path, [*SEQ_EDGE_FIELDS, *STRUCT_EDGE_FIELDS[2:]], rows)


def _write_final_clusters(
    path: Path,
    known_ids: set[str],
    parent: dict[str, str],
    lengths: dict[str, int],
    passing_edges: list[dict[str, str | float | int]],
    sequence_component_by_item: dict[str, str],
    structure_cluster_by_item: dict[str, str],
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
            "sequence_component",
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
                        "sequence_component": sequence_component_by_item.get(item_id, item_id),
                        "structure_cluster": structure_cluster_by_item.get(item_id, item_id),
                        "sequence_length": lengths[item_id],
                    }
                )


def _warn_unmatched(label: str, values: set[str]) -> None:
    values = {value for value in values if value}
    if values:
        preview = ", ".join(sorted(values)[:10])
        warnings.warn(f"unmatched {label} IDs ignored: {preview}", stacklevel=2)


def _reverse(value: str) -> tuple[int, ...]:
    return tuple(-ord(ch) for ch in value)
