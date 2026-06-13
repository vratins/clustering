import sys
from pathlib import Path

from pdbcluster.tools import (
    build_foldseek_multimersearch_cmd,
    build_mmseqs_search_cmd,
    run_command,
)


def test_mmseqs_search_builder_filters_and_sets_exhaustive_max_seqs() -> None:
    cmd = build_mmseqs_search_cmd(
        Path("mmseqs"), Path("chains.fa"), Path("seq.tsv"), Path("tmp"), 0.3, 0.8, 16, True, 42
    )

    assert cmd[:2] == ["mmseqs", "easy-search"]
    assert "query,target,fident,qcov,tcov,alnlen,evalue,bits" in cmd
    assert cmd[cmd.index("--min-seq-id") + 1] == "0.3"
    assert cmd[cmd.index("-c") + 1] == "0.8"
    assert cmd[cmd.index("--max-seqs") + 1] == "42"
    assert "--gpu" in cmd


def test_foldseek_multimersearch_builder_includes_monomer_safe_args() -> None:
    cmd = build_foldseek_multimersearch_cmd(
        Path("foldseek"),
        Path("structures"),
        Path("structures"),
        Path("search"),
        Path("tmp"),
        0.5,
        0.8,
        16,
        True,
        7,
    )

    assert cmd[:6] == [
        "foldseek",
        "easy-multimersearch",
        "structures",
        "structures",
        "search",
        "tmp",
    ]
    assert cmd[cmd.index("--multimer-tm-threshold") + 1] == "0.5"
    assert cmd[cmd.index("-c") + 1] == "0.8"
    assert cmd[cmd.index("--monomer-include-mode") + 1] == "0"
    assert cmd[cmd.index("--min-aligned-chains") + 1] == "1"
    assert cmd[cmd.index("--max-seqs") + 1] == "7"
    assert cmd[cmd.index("--threads") + 1] == "16"
    assert "--gpu" in cmd


def test_run_command_writes_tool_output_and_exit_metadata(tmp_path: Path) -> None:
    log_path = tmp_path / "tool.log"

    run_command([sys.executable, "-c", "print('tool-output')"], log_path)

    log = log_path.read_text()
    assert "tool-output" in log
    assert "exit_code=0" in log
    assert "elapsed_seconds=" in log
