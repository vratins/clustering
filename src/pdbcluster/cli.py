from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .discovery import discover_entries
from .prepare import prepare_inputs
from .tools import bootstrap_tools, nvidia_smi_summary, resolve_tool, tool_version
from .workflows import run_pipeline, split_clusters

app = typer.Typer(no_args_is_help=True, help="Cluster PDB/mmCIF files by sequence and structure.")
console = Console()


@app.command("bootstrap-tools")
def bootstrap_tools_cmd(
    tool_dir: Annotated[Path, typer.Option(help="Project-local tool install directory.")] = Path(
        ".tools"
    ),
    force: Annotated[
        bool, typer.Option(help="Redownload and replace existing tool installs.")
    ] = False,
) -> None:
    """Download project-local GPU-capable MMseqs and Foldseek binaries."""
    results = bootstrap_tools(tool_dir=tool_dir, force=force)
    console.print(json.dumps(results, indent=2, sort_keys=True))


@app.command("validate-tools")
def validate_tools_cmd(
    tool_dir: Annotated[Path, typer.Option(help="Project-local tool install directory.")] = Path(
        ".tools"
    ),
    mmseqs_path: Annotated[Path | None, typer.Option(help="Explicit mmseqs binary path.")] = None,
    foldseek_path: Annotated[
        Path | None, typer.Option(help="Explicit foldseek binary path.")
    ] = None,
) -> None:
    """Check MMseqs, Foldseek, and GPU visibility."""
    mmseqs = resolve_tool("mmseqs", mmseqs_path, tool_dir)
    foldseek = resolve_tool("foldseek", foldseek_path, tool_dir)

    table = Table(title="Tool validation")
    table.add_column("Tool")
    table.add_column("Path")
    table.add_column("Version")
    table.add_row("mmseqs", str(mmseqs or "missing"), tool_version(mmseqs) if mmseqs else "")
    table.add_row(
        "foldseek", str(foldseek or "missing"), tool_version(foldseek) if foldseek else ""
    )
    console.print(table)
    console.print(nvidia_smi_summary())

    if mmseqs is None or foldseek is None:
        raise typer.Exit(code=1)


@app.command("prepare")
def prepare_cmd(
    data_dir: Annotated[
        Path,
        typer.Option(help="Input directory: <data_dir>/<pdb_id>/<pdb_id>_final.cif|pdb."),
    ],
    out_dir: Annotated[Path, typer.Option(help="Output directory for prepared inputs.")],
) -> None:
    """Prepare sequence FASTA and native structure symlink directory."""
    entries = discover_entries(data_dir)
    prepared = prepare_inputs(entries, out_dir)
    console.print(
        f"Prepared {len(prepared.entries)} entries / {len(prepared.chains)} chains: "
        f"{prepared.fasta_path} and {prepared.structures_dir}"
    )


@app.command("run")
def run_cmd(
    data_dir: Annotated[
        Path,
        typer.Option(help="Input directory: <data_dir>/<pdb_id>/<pdb_id>_final.cif|pdb."),
    ],
    out_dir: Annotated[Path, typer.Option(help="Output directory.")],
    seq_id: Annotated[
        float, typer.Option(help="Minimum sequence identity for final fused edges.")
    ] = 0.30,
    seq_cov: Annotated[
        float, typer.Option(help="Minimum query and target sequence coverage.")
    ] = 0.80,
    tm: Annotated[
        float, typer.Option(help="Minimum min(query TM, target TM) for final fused edges.")
    ] = 0.50,
    struct_cov: Annotated[
        float, typer.Option(help="Minimum query and target structure coverage.")
    ] = 0.80,
    complex_seq_cov: Annotated[
        float,
        typer.Option(
            help=(
                "Minimum fraction of each PDB entry covered by optimally matched "
                "non-overlapping sequence chains before Foldseek gating."
            )
        ),
    ] = 0.50,
    threads: Annotated[int, typer.Option(help="Threads for MMseqs/Foldseek.")] = 8,
    max_seqs: Annotated[
        int,
        typer.Option(
            help=(
                "Max results per query passing the prefilter, for both tools. 0 means "
                "never truncate (= all chains for MMseqs, = component size for Foldseek)."
            )
        ),
    ] = 0,
    force: Annotated[
        bool,
        typer.Option(help="Ignore cached stage outputs and recompute everything."),
    ] = False,
    gpu_devices: Annotated[
        str | None,
        typer.Option(help="CUDA_VISIBLE_DEVICES value, e.g. '0' or '0,1,2,3'."),
    ] = None,
    no_gpu: Annotated[
        bool, typer.Option(help="Disable --gpu for MMseqs/Foldseek searches.")
    ] = False,
    tool_dir: Annotated[Path, typer.Option(help="Project-local tool install directory.")] = Path(
        ".tools"
    ),
    mmseqs_path: Annotated[Path | None, typer.Option(help="Explicit mmseqs binary path.")] = None,
    foldseek_path: Annotated[
        Path | None, typer.Option(help="Explicit foldseek binary path.")
    ] = None,
) -> None:
    """Run the full sequence + structure clustering pipeline."""
    run_pipeline(
        data_dir=data_dir,
        out_dir=out_dir,
        seq_id=seq_id,
        seq_cov=seq_cov,
        tm_threshold=tm,
        struct_cov=struct_cov,
        complex_seq_cov=complex_seq_cov,
        threads=threads,
        gpu_devices=gpu_devices,
        tool_dir=tool_dir,
        mmseqs_path=mmseqs_path,
        foldseek_path=foldseek_path,
        use_gpu=not no_gpu,
        max_seqs=max_seqs,
        force=force,
    )
    console.print(f"Wrote clustering outputs to {out_dir}")


@app.command("split")
def split_cmd(
    out_dir: Annotated[
        Path, typer.Option(help="Output directory containing final_clusters.tsv.")
    ],
    split_name: Annotated[str, typer.Option(help="Prefix for output split files.")],
    train: Annotated[float, typer.Option(help="Relative weight for training split.")] = 0.8,
    valid: Annotated[
        float, typer.Option(help="Relative weight for validation split.")
    ] = 0.1,
    test: Annotated[float, typer.Option(help="Relative weight for test split.")] = 0.1,
    seed: Annotated[
        int | None, typer.Option(help="Random seed for reproducible shuffling.")
    ] = None,
    max_cluster_size: Annotated[
        int,
        typer.Option(
            help="Clusters with >= this many members are forced into train only."
        ),
    ] = 500,
) -> None:
    """Create train/valid/test split files from final cluster assignments.

    Reads <out-dir>/final_clusters.tsv and writes three text files:
    <out-dir>/<split-name>_train.txt, _valid.txt, _test.txt.
    Each line is a PDB identifier of the form <pdb_id>_final.
    Splitting is cluster-aware so all members of a cluster land in the same file.
    """
    counts = split_clusters(
        out_dir,
        split_name,
        train=train,
        valid=valid,
        test=test,
        seed=seed,
        max_cluster_size=max_cluster_size,
    )
    console.print(
        f"train={counts['train']}  valid={counts['valid']}  test={counts['test']}"
    )
