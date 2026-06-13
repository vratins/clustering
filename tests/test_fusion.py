from pathlib import Path

import pytest

from pdbcluster.fusion import (
    FusionThresholds,
    collapse_sequence_edges,
    fuse_edges,
    load_cluster_assignments,
    parse_multimer_cluster_report,
    sequence_components,
    write_sequence_edges,
    write_structure_edges,
)
from pdbcluster.prepare import PreparedChain, PreparedEntry


def _entry(pdb_id: str, length: int) -> PreparedEntry:
    chain = PreparedChain(
        chain_uid=f"{pdb_id}__chain0001",
        pdb_id=pdb_id,
        chain_id="A",
        sequence="A" * length,
        sequence_length=length,
    )
    return PreparedEntry(
        pdb_id=pdb_id,
        source_path=Path(f"/data/{pdb_id}/{pdb_id}_final.cif"),
        structure_path=Path(f"/out/work/structures/{pdb_id}.cif"),
        chains=(chain,),
        sequence_length=length,
    )


def test_load_cluster_assignments_maps_members_to_representatives(tmp_path: Path) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("a\tb\nc\tc\n")

    assert load_cluster_assignments([path], {"a", "b", "c"}) == {
        "a": "a",
        "b": "a",
        "c": "c",
    }


def test_sequence_edges_collapse_chain_evidence_to_structure_pairs(tmp_path: Path) -> None:
    seq = tmp_path / "seq.tsv"
    seq.write_text(
        "\n".join(
            [
                "a__chain0001\tb__chain0001\t0.7\t0.9\t0.9\t100\t1e-20\t80",
                "a__chain0002\tb__chain0002\t0.8\t0.9\t0.9\t100\t1e-20\t90",
                "a__chain0001\tc__chain0001\t0.2\t0.9\t0.9\t100\t1e-20\t50",
            ]
        )
        + "\n"
    )

    edges = collapse_sequence_edges(
        seq,
        {
            "a__chain0001": "a",
            "a__chain0002": "a",
            "b__chain0001": "b",
            "b__chain0002": "b",
            "c__chain0001": "c",
        },
        FusionThresholds(seq_id=0.3, seq_cov=0.8, tm=0.5, struct_cov=0.8),
    )

    assert set(edges) == {("a", "b")}
    assert edges[("a", "b")]["seq_identity"] == 0.8
    assert edges[("a", "b")]["seq_evidence_count"] == 2


def test_parse_multimer_cluster_report_reads_documented_order(tmp_path: Path) -> None:
    report = tmp_path / "structure_cluster_cluster_report"
    report.write_text(
        "a\tb\t0.91\t0.92\t0.71\t0.72\t0.80\t1,0,0,0,1,0,0,0,1\t0,0,0\n"
        "b\tc\t0.50\t0.50\t0.20\t0.20\t0.10\t1,0,0,0,1,0,0,0,1\t0,0,0\n"
    )

    edges = parse_multimer_cluster_report(report, {"a", "b", "c"}, "S000001")

    assert edges[("a", "b")]["complex_qtm"] == 0.71
    assert edges[("a", "b")]["complex_tcov"] == 0.92
    assert edges[("a", "b")]["source_component"] == "S000001"


def test_fuse_edges_requires_sequence_and_structure_thresholds(tmp_path: Path) -> None:
    seq_edges = {
        ("a", "b"): {
            "item_a": "a",
            "item_b": "b",
            "seq_identity": 0.7,
            "seq_qcov": 0.9,
            "seq_tcov": 0.9,
            "seq_query_chain": "a__chain0001",
            "seq_target_chain": "b__chain0001",
            "seq_evidence_count": 1,
        },
        ("b", "c"): {
            "item_a": "b",
            "item_b": "c",
            "seq_identity": 0.7,
            "seq_qcov": 0.9,
            "seq_tcov": 0.9,
            "seq_query_chain": "b__chain0001",
            "seq_target_chain": "c__chain0001",
            "seq_evidence_count": 1,
        },
    }
    struct_edges = {
        ("a", "b"): {
            "item_a": "a",
            "item_b": "b",
            "complex_qtm": 0.8,
            "complex_ttm": 0.75,
            "complex_qcov": 0.9,
            "complex_tcov": 0.9,
            "interface_lddt": 0.8,
            "source_component": "S000001",
        },
        ("b", "c"): {
            "item_a": "b",
            "item_b": "c",
            "complex_qtm": 0.8,
            "complex_ttm": 0.30,
            "complex_qcov": 0.9,
            "complex_tcov": 0.9,
            "interface_lddt": 0.8,
            "source_component": "S000001",
        },
    }
    seq = tmp_path / "seq.tsv"
    struct = tmp_path / "struct.tsv"
    write_sequence_edges(seq, seq_edges)
    write_structure_edges(struct, struct_edges)

    fuse_edges(
        entries=[_entry("a", 100), _entry("b", 120), _entry("c", 140)],
        seq_edges_path=seq,
        struct_edges_path=struct,
        out_dir=tmp_path,
        thresholds=FusionThresholds(seq_id=0.3, seq_cov=0.8, tm=0.5, struct_cov=0.8),
        sequence_component_by_item={"a": "S000001", "b": "S000001", "c": "S000001"},
        structure_cluster_by_item={"a": "a", "b": "a", "c": "c"},
    )

    final_edges = (tmp_path / "final_edges.tsv").read_text()
    final_clusters = (tmp_path / "final_clusters.tsv").read_text()
    assert "a\tb" in final_edges
    assert "b\tc" not in final_edges
    assert "complex_qtm" in final_edges
    assert "a\tC000001\tb\tS000001\ta" in final_clusters
    assert "b\tC000001\tb\tS000001\ta" in final_clusters
    assert "c\tC000002\tc\tS000001\tc" in final_clusters


def test_sequence_components_keep_disconnected_groups_separate() -> None:
    components = sequence_components(
        {"a", "b", "c", "d"},
        {
            ("a", "b"): {"item_a": "a", "item_b": "b"},
            ("c", "d"): {"item_a": "c", "item_b": "d"},
        },
    )

    assert components["a"] == components["b"]
    assert components["c"] == components["d"]
    assert components["a"] != components["c"]


def test_unmatched_ids_warn_loudly(tmp_path: Path) -> None:
    seq = tmp_path / "seq.tsv"
    seq.write_text("missing\ta__chain0001\t0.7\t0.9\t0.9\t100\t1e-20\t80\n")

    with pytest.warns(UserWarning, match="unmatched MMseqs chain IDs"):
        collapse_sequence_edges(
            seq,
            {"a__chain0001": "a"},
            FusionThresholds(seq_id=0.3, seq_cov=0.8, tm=0.5, struct_cov=0.8),
        )
