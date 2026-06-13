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
from .workflows import run_pipeline

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
    threads: Annotated[int, typer.Option(help="Threads for MMseqs/Foldseek.")] = 8,
    max_seqs: Annotated[
        int,
        typer.Option(
            help=(
                "MMseqs/Foldseek --max-seqs. 0 resolves to all chains for "
                "MMseqs and component size for Foldseek."
            )
        ),
    ] = 0,
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
        threads=threads,
        gpu_devices=gpu_devices,
        tool_dir=tool_dir,
        mmseqs_path=mmseqs_path,
        foldseek_path=foldseek_path,
        use_gpu=not no_gpu,
        max_seqs=max_seqs,
    )
    console.print(f"Wrote clustering outputs to {out_dir}")
