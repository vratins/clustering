from pathlib import Path

from pdbcluster.tools import (
    build_foldseek_cluster_cmd,
    build_foldseek_search_cmd,
    build_mmseqs_cluster_cmd,
    build_mmseqs_search_cmd,
)


def test_mmseqs_command_builders_include_thresholds_and_gpu_flag() -> None:
    cluster = build_mmseqs_cluster_cmd(
        Path("mmseqs"), Path("seq.fa"), Path("seq_cluster"), Path("tmp"), 0.3, 0.8, 16
    )
    search = build_mmseqs_search_cmd(
        Path("mmseqs"), Path("seq.fa"), Path("seq.tsv"), Path("tmp"), 16, True
    )

    assert cluster[:2] == ["mmseqs", "easy-cluster"]
    assert "--min-seq-id" in cluster
    assert "--gpu" in search
    assert "query,target,fident,qcov,tcov,alnlen,evalue,bits" in search


def test_foldseek_command_builders_use_native_structure_directory() -> None:
    cluster = build_foldseek_cluster_cmd(
        Path("foldseek"),
        Path("structures"),
        Path("structure_cluster"),
        Path("tmp"),
        0.5,
        0.8,
        16,
        True,
    )
    search = build_foldseek_search_cmd(
        Path("foldseek"), Path("structures"), Path("structure.tsv"), Path("tmp"), 16, True
    )

    assert cluster[:3] == ["foldseek", "easy-cluster", "structures"]
    assert "--tmscore-threshold" in cluster
    assert search[:3] == ["foldseek", "easy-search", "structures"]
    assert "--gpu" in search
