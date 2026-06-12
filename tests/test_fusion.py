from pathlib import Path

from pdbcluster.fusion import FusionThresholds, fuse_edges, load_cluster_assignments
from pdbcluster.prepare import PreparedEntry


def _entry(pdb_id: str, length: int) -> PreparedEntry:
    return PreparedEntry(
        pdb_id=pdb_id,
        source_path=Path(f"/data/{pdb_id}/{pdb_id}_final.cif"),
        structure_path=Path(f"/out/work/structures/{pdb_id}.cif"),
        sequence="A" * length,
        sequence_length=length,
        chain_id="A",
    )


def test_load_cluster_assignments_maps_members_to_representatives(tmp_path: Path) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("a\tb\nc\tc\n")

    assert load_cluster_assignments(path, {"a", "b", "c"}) == {"a": "a", "b": "a", "c": "c"}


def test_fuse_edges_requires_sequence_and_structure_thresholds(tmp_path: Path) -> None:
    seq = tmp_path / "seq.tsv"
    struct = tmp_path / "struct.tsv"
    seq_clu = tmp_path / "seq_cluster.tsv"
    struct_clu = tmp_path / "struct_cluster.tsv"
    seq.write_text(
        "\n".join(
            [
                "a\tb\t0.7\t0.9\t0.9\t100\t1e-20\t80",
                "b\tc\t0.7\t0.9\t0.9\t100\t1e-20\t80",
            ]
        )
        + "\n"
    )
    struct.write_text(
        "\n".join(
            [
                "a\tb\t0.7\t0.8\t0.75\t0.8\t0.9\t0.4\t0.9\t0.9\t100\t1e-20\t80",
                "b\tc\t0.7\t0.8\t0.30\t0.8\t0.9\t0.4\t0.9\t0.9\t100\t1e-20\t80",
            ]
        )
        + "\n"
    )
    seq_clu.write_text("a\ta\na\tb\nc\tc\n")
    struct_clu.write_text("a\ta\na\tb\nc\tc\n")

    fuse_edges(
        entries=[_entry("a", 100), _entry("b", 120), _entry("c", 140)],
        seq_edges_path=seq,
        struct_edges_path=struct,
        seq_clusters_path=seq_clu,
        struct_clusters_path=struct_clu,
        out_dir=tmp_path,
        thresholds=FusionThresholds(seq_id=0.3, seq_cov=0.8, tm=0.5, struct_cov=0.8),
    )

    final_edges = (tmp_path / "final_edges.tsv").read_text()
    final_clusters = (tmp_path / "final_clusters.tsv").read_text()
    assert "a\tb" in final_edges
    assert "b\tc" not in final_edges
    assert "a\tC000001\tb" in final_clusters
    assert "b\tC000001\tb" in final_clusters
    assert "c\tC000002\tc" in final_clusters
