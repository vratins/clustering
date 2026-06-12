from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .discovery import discover_entries
from .fusion import FusionThresholds, find_cluster_tsv, fuse_edges
from .prepare import PreparedInputs, prepare_inputs
from .tools import (
    ToolPaths,
    build_foldseek_cluster_cmd,
    build_foldseek_search_cmd,
    build_mmseqs_cluster_cmd,
    build_mmseqs_search_cmd,
    resolve_tools,
    run_command,
    tool_version,
)


def run_pipeline(
    data_dir: Path,
    out_dir: Path,
    seq_id: float,
    seq_cov: float,
    tm_threshold: float,
    struct_cov: float,
    threads: int,
    gpu_devices: str | None,
    tool_dir: Path,
    mmseqs_path: Path | None,
    foldseek_path: Path | None,
    use_gpu: bool,
) -> None:
    started = time.time()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_entries(data_dir)
    if not entries:
        raise RuntimeError(f"no entries found in {data_dir}")

    prepared = prepare_inputs(entries, out_dir)
    tools = resolve_tools(tool_dir, mmseqs_path, foldseek_path)
    env = os.environ.copy()
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices

    command_log: list[list[str]] = []
    _run_tool_workflows(
        prepared,
        out_dir,
        tools,
        seq_id,
        seq_cov,
        tm_threshold,
        struct_cov,
        threads,
        use_gpu,
        env,
        command_log,
    )

    seq_cluster_prefix = out_dir / "mmseqs" / "sequence_cluster"
    struct_cluster_prefix = out_dir / "foldseek" / "structure_cluster"
    fuse_edges(
        entries=prepared.entries,
        seq_edges_path=out_dir / "mmseqs" / "seq_edges.tsv",
        struct_edges_path=out_dir / "foldseek" / "structure_edges.tsv",
        seq_clusters_path=find_cluster_tsv(seq_cluster_prefix),
        struct_clusters_path=find_cluster_tsv(struct_cluster_prefix),
        out_dir=out_dir,
        thresholds=FusionThresholds(
            seq_id=seq_id,
            seq_cov=seq_cov,
            tm=tm_threshold,
            struct_cov=struct_cov,
        ),
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
            "threads": threads,
            "gpu_devices": gpu_devices or "",
            "use_gpu": use_gpu,
        },
    )


def _run_tool_workflows(
    prepared: PreparedInputs,
    out_dir: Path,
    tools: ToolPaths,
    seq_id: float,
    seq_cov: float,
    tm_threshold: float,
    struct_cov: float,
    threads: int,
    use_gpu: bool,
    env: dict[str, str],
    command_log: list[list[str]],
) -> None:
    mmseqs_dir = out_dir / "mmseqs"
    foldseek_dir = out_dir / "foldseek"
    mmseqs_dir.mkdir(exist_ok=True)
    foldseek_dir.mkdir(exist_ok=True)

    commands = [
        (
            build_mmseqs_cluster_cmd(
                tools.mmseqs,
                prepared.fasta_path,
                mmseqs_dir / "sequence_cluster",
                mmseqs_dir / "tmp_cluster",
                seq_id,
                seq_cov,
                threads,
            ),
            mmseqs_dir / "sequence_cluster.log",
        ),
        (
            build_mmseqs_search_cmd(
                tools.mmseqs,
                prepared.fasta_path,
                mmseqs_dir / "seq_edges.tsv",
                mmseqs_dir / "tmp_search",
                threads,
                use_gpu,
            ),
            mmseqs_dir / "seq_edges.log",
        ),
        (
            build_foldseek_cluster_cmd(
                tools.foldseek,
                prepared.structures_dir,
                foldseek_dir / "structure_cluster",
                foldseek_dir / "tmp_cluster",
                tm_threshold,
                struct_cov,
                threads,
                use_gpu,
            ),
            foldseek_dir / "structure_cluster.log",
        ),
        (
            build_foldseek_search_cmd(
                tools.foldseek,
                prepared.structures_dir,
                foldseek_dir / "structure_edges.tsv",
                foldseek_dir / "tmp_search",
                threads,
                use_gpu,
            ),
            foldseek_dir / "structure_edges.log",
        ),
    ]

    for cmd, log_path in commands:
        command_log.append(cmd)
        run_command(cmd, log_path, env=env)


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
        "counts": {"prepared_entries": len(prepared.entries)},
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
