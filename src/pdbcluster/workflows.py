from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from .discovery import discover_entries
from .fusion import (
    FusionThresholds,
    collapse_sequence_edges,
    find_cluster_report,
    find_cluster_tsv,
    fuse_edges,
    load_cluster_assignments,
    parse_multimer_cluster_report,
    sequence_components,
    write_sequence_edges,
    write_structure_edges,
)
from .prepare import PreparedEntry, PreparedInputs, prepare_inputs
from .tools import (
    ToolPaths,
    build_foldseek_multimer_report_cmd,
    build_foldseek_multimercluster_cmd,
    build_mmseqs_search_cmd,
    resolve_tools,
    run_command,
    tool_version,
)


class ProgressTracker:
    def __init__(self, path: Path, echo: bool = False) -> None:
        self.path = path
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def event(
        self,
        stage: str,
        status: str,
        message: str = "",
        log_path: Path | None = None,
        **extra: object,
    ) -> None:
        timestamp = time.time()
        row: dict[str, object] = {
            "time": round(timestamp, 3),
            "stage": stage,
            "status": status,
        }
        if message:
            row["message"] = message
        if log_path is not None:
            row["log_path"] = str(log_path)
        row.update(extra)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if self.echo:
            print(_format_progress_line(row, timestamp), file=sys.stderr, flush=True)


def _format_progress_line(row: dict[str, object], timestamp: float) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    stage = str(row["stage"])
    status = str(row["status"])
    message = str(row.get("message", ""))
    parts = [f"[{stamp}]", f"{stage}:{status}"]
    if message:
        parts.append(message)
    if "component_id" in row:
        parts.append(f"component={row['component_id']}")
    if "log_path" in row:
        parts.append(f"log={row['log_path']}")
    return " ".join(parts)


def run_pipeline(
    data_dir: Path,
    out_dir: Path,
    seq_id: float,
    seq_cov: float,
    tm_threshold: float,
    struct_cov: float,
    complex_seq_cov: float,
    threads: int,
    gpu_devices: str | None,
    tool_dir: Path,
    mmseqs_path: Path | None,
    foldseek_path: Path | None,
    use_gpu: bool,
    max_seqs: int,
    foldseek_max_seqs: int,
) -> None:
    started = time.time()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(out_dir / "progress.jsonl", echo=True)
    progress.event("pipeline", "started", out_dir=str(out_dir))

    try:
        progress.event("discover", "started", str(data_dir))
        entries = discover_entries(data_dir)
        progress.event("discover", "done", entries=len(entries))
        if not entries:
            raise RuntimeError(f"no entries found in {data_dir}")

        progress.event("prepare", "started")
        prepared = prepare_inputs(entries, out_dir)
        progress.event(
            "prepare",
            "cached" if prepared.cached else "done",
            entries=len(prepared.entries),
            chains=len(prepared.chains),
            manifest_path=str(prepared.manifest_path),
        )

        progress.event("resolve_tools", "started")
        tools = resolve_tools(tool_dir, mmseqs_path, foldseek_path)
        progress.event(
            "resolve_tools",
            "done",
            mmseqs=str(tools.mmseqs),
            foldseek=str(tools.foldseek),
        )

        env = os.environ.copy()
        if gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = gpu_devices

        thresholds = FusionThresholds(
            seq_id=seq_id,
            seq_cov=seq_cov,
            tm=tm_threshold,
            struct_cov=struct_cov,
            complex_seq_cov=complex_seq_cov,
        )
        command_log: list[list[str]] = []
        sequence_component_by_item, structure_cluster_by_item = _run_tool_workflows(
            prepared,
            out_dir,
            tools,
            thresholds,
            threads,
            use_gpu,
            max_seqs,
            foldseek_max_seqs,
            env,
            command_log,
            progress,
        )

        progress.event("fusion", "started")
        fuse_edges(
            entries=prepared.entries,
            seq_edges_path=out_dir / "mmseqs" / "structure_seq_edges.tsv",
            struct_edges_path=out_dir / "foldseek" / "structure_edges.tsv",
            out_dir=out_dir,
            thresholds=thresholds,
            sequence_component_by_item=sequence_component_by_item,
            structure_cluster_by_item=structure_cluster_by_item,
        )
        progress.event(
            "fusion",
            "done",
            final_edges=str(out_dir / "final_edges.tsv"),
            final_clusters=str(out_dir / "final_clusters.tsv"),
        )

        _write_run_manifest(
            out_dir,
            data_dir,
            prepared,
            tools,
            command_log,
            started,
            {
                "seq_id": seq_id,
                "seq_cov": seq_cov,
                "tm": tm_threshold,
                "struct_cov": struct_cov,
                "complex_seq_cov": complex_seq_cov,
                "threads": threads,
                "gpu_devices": gpu_devices or "",
                "use_gpu": use_gpu,
                "max_seqs": max_seqs,
                "foldseek_max_seqs": foldseek_max_seqs,
                "progress_path": str(progress.path),
            },
        )
        progress.event(
            "pipeline",
            "done",
            elapsed_seconds=round(time.time() - started, 3),
        )
    except Exception as exc:
        progress.event("pipeline", "failed", str(exc))
        raise


def _run_tool_workflows(
    prepared: PreparedInputs,
    out_dir: Path,
    tools: ToolPaths,
    thresholds: FusionThresholds,
    threads: int,
    use_gpu: bool,
    max_seqs: int,
    foldseek_max_seqs: int,
    env: dict[str, str],
    command_log: list[list[str]],
    progress: ProgressTracker,
) -> tuple[dict[str, str], dict[str, str]]:
    mmseqs_dir = out_dir / "mmseqs"
    foldseek_dir = out_dir / "foldseek"
    mmseqs_dir.mkdir(exist_ok=True)
    foldseek_dir.mkdir(exist_ok=True)

    mmseqs_max = max_seqs if max_seqs > 0 else len(prepared.chains)
    seq_edges_path = mmseqs_dir / "seq_edges.tsv"
    mmseqs_cmd = build_mmseqs_search_cmd(
        tools.mmseqs,
        prepared.fasta_path,
        seq_edges_path,
        mmseqs_dir / "tmp_search",
        thresholds.seq_id,
        thresholds.seq_cov,
        threads,
        use_gpu,
        mmseqs_max,
    )
    _run_cached_command(
        "mmseqs_chain_search",
        f"chain all-vs-all search over {len(prepared.chains)} chains",
        mmseqs_cmd,
        mmseqs_dir / "seq_edges.log",
        mmseqs_dir / "seq_edges.params.json",
        [seq_edges_path],
        {
            "stage": "mmseqs_chain_search",
            "command": mmseqs_cmd,
            "tool_version": tool_version(tools.mmseqs),
            "input_fingerprint": _fingerprint_paths([prepared.fasta_path]),
        },
        env,
        command_log,
        progress,
        max_seqs=mmseqs_max,
    )

    progress.event("sequence_edge_collapse", "started")
    chain_to_structure = {chain.chain_uid: chain.pdb_id for chain in prepared.chains}
    chain_lengths = {chain.chain_uid: chain.sequence_length for chain in prepared.chains}
    structure_lengths = {entry.pdb_id: entry.sequence_length for entry in prepared.entries}
    seq_edges = collapse_sequence_edges(
        seq_edges_path,
        chain_to_structure,
        thresholds,
        chain_lengths=chain_lengths,
        structure_lengths=structure_lengths,
    )
    structure_seq_edges_path = mmseqs_dir / "structure_seq_edges.tsv"
    write_sequence_edges(structure_seq_edges_path, seq_edges)

    known_ids = {entry.pdb_id for entry in prepared.entries}
    sequence_component_by_item = sequence_components(known_ids, seq_edges)
    members_by_component: dict[str, list[str]] = defaultdict(list)
    for item_id, component_id in sequence_component_by_item.items():
        members_by_component[component_id].append(item_id)
    gated_components = sum(1 for members in members_by_component.values() if len(members) > 1)
    progress.event(
        "sequence_edge_collapse",
        "done",
        structure_edges=len(seq_edges),
        sequence_components=len(members_by_component),
        foldseek_components=gated_components,
        output_path=str(structure_seq_edges_path),
    )

    entries_by_id = {entry.pdb_id: entry for entry in prepared.entries}
    all_structure_edges: dict[tuple[str, str], dict[str, str | float]] = {}
    cluster_paths: list[Path] = []
    for component_id, members in sorted(members_by_component.items()):
        if len(members) < 2:
            progress.event(
                "foldseek_multimercluster",
                "skipped",
                "singleton sequence component",
                component_id=component_id,
                members=sorted(members),
            )
            continue
        component_entries = [entries_by_id[item_id] for item_id in sorted(members)]
        component_dir = foldseek_dir / "components" / component_id
        component_structures = component_dir / "structures"
        _prepare_component_structures(component_entries, component_structures)

        foldseek_max = foldseek_max_seqs if foldseek_max_seqs > 0 else 300
        cluster_prefix = component_dir / "structure_cluster"
        report_path = cluster_prefix.with_name(cluster_prefix.name + "_cluster_report")
        cluster_path = cluster_prefix.with_name(cluster_prefix.name + "_cluster.tsv")
        cmd = build_foldseek_multimercluster_cmd(
            tools.foldseek,
            component_structures,
            cluster_prefix,
            component_dir / "tmp",
            thresholds.tm,
            thresholds.struct_cov,
            threads,
            use_gpu,
            foldseek_max,
        )
        _run_cached_command(
            "foldseek_multimercluster",
            f"component {component_id} over {len(component_entries)} structures",
            cmd,
            component_dir / "structure_cluster.log",
            component_dir / "structure_cluster.params.json",
            [report_path, cluster_path],
            {
                "stage": "foldseek_multimercluster",
                "component_id": component_id,
                "members": sorted(members),
                "command": cmd,
                "tool_version": tool_version(tools.foldseek),
                "input_fingerprint": _fingerprint_paths(
                    [entry.structure_path for entry in component_entries]
                ),
            },
            env,
            command_log,
            progress,
            component_id=component_id,
            members=sorted(members),
            max_seqs=foldseek_max,
        )

        actual_report = _ensure_multimer_report(
            tools,
            cluster_prefix,
            component_dir / "tmp",
            component_dir / "structure_cluster_report.log",
            component_dir / "structure_cluster_report.params.json",
            component_id,
            threads,
            env,
            command_log,
            progress,
        )
        for pair, edge in parse_multimer_cluster_report(
            actual_report, known_ids, component_id, thresholds.struct_cov
        ).items():
            previous = all_structure_edges.get(pair)
            score = min(float(edge["complex_qtm"]), float(edge["complex_ttm"]))
            previous_score = (
                min(float(previous["complex_qtm"]), float(previous["complex_ttm"]))
                if previous
                else -1.0
            )
            if score > previous_score:
                all_structure_edges[pair] = edge

        actual_cluster = find_cluster_tsv(cluster_prefix)
        if actual_cluster.exists():
            cluster_paths.append(actual_cluster)
        shutil.rmtree(component_dir / "tmp", ignore_errors=True)

    structure_edges_path = foldseek_dir / "structure_edges.tsv"
    write_structure_edges(structure_edges_path, all_structure_edges)
    progress.event(
        "structure_edge_merge",
        "done",
        structure_edges=len(all_structure_edges),
        output_path=str(structure_edges_path),
    )
    structure_cluster_by_item = load_cluster_assignments(cluster_paths, known_ids)
    return sequence_component_by_item, structure_cluster_by_item


def _ensure_multimer_report(
    tools: ToolPaths,
    cluster_prefix: Path,
    tmp_dir: Path,
    log_path: Path,
    params_path: Path,
    component_id: str,
    threads: int,
    env: dict[str, str],
    command_log: list[list[str]],
    progress: ProgressTracker,
) -> Path:
    actual_report = find_cluster_report(cluster_prefix)
    if actual_report.exists():
        return actual_report

    report_path = cluster_prefix.with_name(cluster_prefix.name + "_cluster_report")
    report_inputs = _find_multimer_report_inputs(tmp_dir)
    if report_inputs is None:
        raise RuntimeError(
            f"Foldseek did not write a multimer cluster report for {component_id}, "
            f"and retained report DBs were not found under {tmp_dir}: expected {report_path}"
        )

    query_db, result_db = report_inputs
    cmd = build_foldseek_multimer_report_cmd(
        tools.foldseek,
        query_db,
        result_db,
        report_path,
        threads,
    )
    _run_cached_command(
        "foldseek_multimer_report",
        f"component {component_id} report export",
        cmd,
        log_path,
        params_path,
        [report_path],
        {
            "stage": "foldseek_multimer_report",
            "component_id": component_id,
            "command": cmd,
            "tool_version": tool_version(tools.foldseek),
        },
        env,
        command_log,
        progress,
        component_id=component_id,
    )

    actual_report = find_cluster_report(cluster_prefix)
    if not actual_report.exists():
        raise RuntimeError(
            f"Foldseek report export did not write a multimer cluster report for {component_id}: "
            f"expected {actual_report}"
        )
    return actual_report


def _find_multimer_report_inputs(tmp_dir: Path) -> tuple[Path, Path] | None:
    if not tmp_dir.exists():
        return None

    candidates: list[tuple[int, Path, Path]] = []
    for work_dir in tmp_dir.iterdir():
        if not work_dir.is_dir():
            continue
        query_candidates = [work_dir / "query_pad", work_dir / "query"]
        result_dbtypes = sorted(work_dir.glob("multimercluster_tmp/*/multimer_result.dbtype"))
        for query_db in query_candidates:
            query_dbtype = Path(str(query_db) + ".dbtype")
            if not query_dbtype.exists():
                continue
            for result_dbtype in result_dbtypes:
                result_db = result_dbtype.with_suffix("")
                mtime = max(query_dbtype.stat().st_mtime_ns, result_dbtype.stat().st_mtime_ns)
                candidates.append((mtime, query_db, result_db))

    if not candidates:
        return None
    _mtime, query_db, result_db = max(candidates, key=lambda item: item[0])
    return query_db, result_db


def _run_cached_command(
    stage: str,
    message: str,
    cmd: list[str],
    log_path: Path,
    params_path: Path,
    output_paths: list[Path],
    params: dict[str, object],
    env: dict[str, str],
    command_log: list[list[str]],
    progress: ProgressTracker,
    **extra: object,
) -> None:
    if _cache_valid(params_path, output_paths, params):
        progress.event(stage, "cached", message, log_path, **extra)
        return

    progress.event(stage, "started", message, log_path, **extra)
    command_log.append(cmd)
    try:
        run_command(cmd, log_path, env=env)
    except Exception as exc:
        progress.event(stage, "failed", str(exc), log_path, **extra)
        raise
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    progress.event(stage, "done", message, log_path, **extra)


def _cache_valid(params_path: Path, output_paths: list[Path], params: dict[str, object]) -> bool:
    if not params_path.exists() or any(not path.exists() for path in output_paths):
        return False
    try:
        cached = json.loads(params_path.read_text())
    except json.JSONDecodeError:
        return False
    return cached == params


def _fingerprint_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _prepare_component_structures(entries: list[PreparedEntry], structures_dir: Path) -> None:
    structures_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{entry.pdb_id}{entry.structure_path.suffix.lower()}" for entry in entries}
    for path in structures_dir.iterdir():
        if path.name not in expected:
            path.unlink()
    for entry in entries:
        target = structures_dir / f"{entry.pdb_id}{entry.structure_path.suffix.lower()}"
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(entry.structure_path)
        except OSError:
            shutil.copy2(entry.structure_path, target)


def _write_run_manifest(
    out_dir: Path,
    data_dir: Path,
    prepared: PreparedInputs,
    tools: ToolPaths,
    command_log: list[list[str]],
    started: float,
    config: dict[str, float | int | str | bool],
) -> None:
    manifest = {
        "data_dir": str(data_dir.expanduser().resolve()),
        "out_dir": str(out_dir),
        "config": config,
        "counts": {
            "prepared_entries": len(prepared.entries),
            "prepared_chains": len(prepared.chains),
        },
        "tools": {
            "mmseqs": {"path": str(tools.mmseqs), "version": tool_version(tools.mmseqs)},
            "foldseek": {"path": str(tools.foldseek), "version": tool_version(tools.foldseek)},
        },
        "commands": command_log,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
