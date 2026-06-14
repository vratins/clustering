import json
from pathlib import Path

import pytest

from pdbcluster.fusion import FusionThresholds
from pdbcluster.prepare import PreparedChain, PreparedEntry, PreparedInputs
from pdbcluster.tools import ToolPaths
from pdbcluster.workflows import ProgressTracker, _cache_valid, _run_tool_workflows, split_clusters

THRESHOLDS = FusionThresholds(seq_id=0.3, seq_cov=0.8, tm=0.5, struct_cov=0.8)
TOOLS = ToolPaths(mmseqs=Path("mmseqs"), foldseek=Path("foldseek"))


def _prepared(tmp_path: Path) -> PreparedInputs:
    structures_dir = tmp_path / "work" / "structures"
    structures_dir.mkdir(parents=True)
    chains: list[PreparedChain] = []
    entries: list[PreparedEntry] = []
    for pdb_id in ["a", "b", "c", "d"]:
        structure = structures_dir / f"{pdb_id}.cif"
        structure.write_text(f"structure {pdb_id}\n")
        chain = PreparedChain(
            chain_uid=f"{pdb_id}__chain0001",
            pdb_id=pdb_id,
            chain_id="A",
            sequence="AAAA",
            sequence_length=4,
        )
        chains.append(chain)
        entries.append(
            PreparedEntry(
                pdb_id=pdb_id,
                source_path=structure,
                structure_path=structure,
                chains=(chain,),
                sequence_length=4,
            )
        )
    fasta = tmp_path / "work" / "chains.fasta"
    fasta.write_text("".join(f">{chain.chain_uid}\n{chain.sequence}\n" for chain in chains))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("manifest\n")
    return PreparedInputs(entries, chains, fasta, structures_dir, manifest)


def _write_seq_edges(output_tsv: str) -> None:
    Path(output_tsv).write_text(
        "a__chain0001\tb__chain0001\t0.9\t1.0\t1.0\t4\t1e-9\t50\n"
        "c__chain0001\td__chain0001\t0.9\t1.0\t1.0\t4\t1e-9\t50\n"
    )


def _write_report(out_prefix: str, structures: list[str]) -> None:
    # 9-col easy-multimersearch report: q t qChains tChains qTM tTM u t assId
    a, b = structures
    report = Path(out_prefix).with_name(Path(out_prefix).name + "_report")
    report.write_text(f"{a}\t{b}\tA\tA\t0.90\t0.85\t1\t0\t0\n")


def test_foldseek_runs_only_inside_sequence_components(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    foldseek_calls: list[list[str]] = []

    monkeypatch.setattr("pdbcluster.workflows.tool_version", lambda _path: "tool-v1")

    def fake_run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ran\n")
        if cmd[1] == "easy-search":
            _write_seq_edges(cmd[4])
            return
        assert cmd[1] == "easy-multimersearch"
        structures = sorted(path.stem for path in Path(cmd[2]).iterdir())
        foldseek_calls.append(structures)
        _write_report(cmd[4], structures)

    monkeypatch.setattr("pdbcluster.workflows.run_command", fake_run_command)

    command_log: list[list[str]] = []
    progress = ProgressTracker(tmp_path / "progress.jsonl")
    seq_components = _run_tool_workflows(
        prepared,
        tmp_path,
        TOOLS,
        THRESHOLDS,
        threads=8,
        use_gpu=False,
        max_seqs=0,
        env={},
        command_log=command_log,
        progress=progress,
    )

    events = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert {event["stage"] for event in events} >= {
        "mmseqs_chain_search",
        "sequence_edge_collapse",
        "foldseek_multimersearch",
        "structure_edge_merge",
    }
    assert any(event["status"] == "done" for event in events)
    assert any("log_path" in event for event in events)
    assert any(event.get("component") == "1/2" for event in events)

    # one Foldseek run per >=2-member sequence component; singletons (none here) skip
    assert foldseek_calls == [["a", "b"], ["c", "d"]]
    assert seq_components["a"] == seq_components["b"]
    assert seq_components["c"] == seq_components["d"]
    assert seq_components["a"] != seq_components["c"]

    # max_seqs=0 -> all chains for MMseqs, component size for Foldseek
    assert command_log[0][command_log[0].index("--max-seqs") + 1] == "4"
    assert [cmd[cmd.index("--max-seqs") + 1] for cmd in command_log[1:]] == ["2", "2"]

    structure_edges = (tmp_path / "foldseek" / "structure_edges.tsv").read_text()
    assert "a\tb\t0.9\t0.85\tNA\tNA\tNA" in structure_edges


def test_singleton_sequence_component_skips_foldseek(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    foldseek_calls: list[list[str]] = []

    monkeypatch.setattr("pdbcluster.workflows.tool_version", lambda _path: "tool-v1")

    def fake_run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ran\n")
        if cmd[1] == "easy-search":
            # only a~b are homologous; c and d are sequence singletons
            Path(cmd[4]).write_text("a__chain0001\tb__chain0001\t0.9\t1.0\t1.0\t4\t1e-9\t50\n")
            return
        assert cmd[1] == "easy-multimersearch"
        structures = sorted(path.stem for path in Path(cmd[2]).iterdir())
        foldseek_calls.append(structures)
        _write_report(cmd[4], structures)

    monkeypatch.setattr("pdbcluster.workflows.run_command", fake_run_command)

    progress = ProgressTracker(tmp_path / "progress.jsonl")
    _run_tool_workflows(
        prepared,
        tmp_path,
        TOOLS,
        THRESHOLDS,
        threads=8,
        use_gpu=False,
        max_seqs=0,
        env={},
        command_log=[],
        progress=progress,
    )

    assert foldseek_calls == [["a", "b"]]


def test_force_recomputes_after_cache_hit(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr("pdbcluster.workflows.tool_version", lambda _path: "tool-v1")

    def fake_run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ran\n")
        calls.append(cmd[1])
        if cmd[1] == "easy-search":
            _write_seq_edges(cmd[4])
            return
        structures = sorted(path.stem for path in Path(cmd[2]).iterdir())
        _write_report(cmd[4], structures)

    monkeypatch.setattr("pdbcluster.workflows.run_command", fake_run_command)

    def run(force: bool) -> None:
        _run_tool_workflows(
            prepared,
            tmp_path,
            TOOLS,
            THRESHOLDS,
            threads=8,
            use_gpu=False,
            max_seqs=0,
            env={},
            command_log=[],
            progress=ProgressTracker(tmp_path / "progress.jsonl"),
            force=force,
        )

    run(force=False)
    assert "easy-search" in calls and "easy-multimersearch" in calls

    calls.clear()
    run(force=False)
    assert calls == []  # every stage served from cache

    calls.clear()
    run(force=True)
    assert "easy-search" in calls and calls.count("easy-multimersearch") == 2


def test_progress_tracker_can_echo_live_events(tmp_path: Path, capsys) -> None:
    progress = ProgressTracker(tmp_path / "progress.jsonl", echo=True)

    progress.event("stage", "started", "working", log_path=tmp_path / "stage.log")

    captured = capsys.readouterr()
    assert "stage:started" in captured.err
    assert "working" in captured.err
    assert "log=" in captured.err
    assert json.loads(progress.path.read_text())["stage"] == "stage"


def test_cache_valid_requires_matching_params_and_outputs(tmp_path: Path) -> None:
    params_path = tmp_path / "stage.params.json"
    output = tmp_path / "out.tsv"
    params = {"command": ["tool"], "threshold": 0.5}
    params_path.write_text('{"command":["tool"],"threshold":0.5}\n')
    output.write_text("ok\n")

    assert _cache_valid(params_path, [output], params)
    assert not _cache_valid(params_path, [output], {"command": ["tool"], "threshold": 0.6})
    assert not _cache_valid(params_path, [tmp_path / "missing.tsv"], params)
    params_path.write_text("not-json\n")
    assert not _cache_valid(params_path, [output], params)


# --- split_clusters tests ---


def _write_clusters_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a minimal final_clusters.tsv with (pdb_id, final_cluster) rows."""
    path.write_text(
        "pdb_id\tfinal_cluster\tfinal_representative\tsequence_component\tsequence_length\n"
        + "".join(f"{pdb}\t{cluster}\trep\tS000001\t100\n" for pdb, cluster in rows)
    )


def test_split_clusters_basic_partition(tmp_path: Path) -> None:
    # 10 singleton clusters; largest-first ordering puts test/valid before train
    rows = [(f"p{i:02d}", f"C{i:06d}") for i in range(10)]
    _write_clusters_tsv(tmp_path / "final_clusters.tsv", rows)

    counts = split_clusters(tmp_path, "mydb", train=0.8, valid=0.1, test=0.1, seed=0)

    train_lines = (tmp_path / "mydb_train.txt").read_text().splitlines()
    valid_lines = (tmp_path / "mydb_valid.txt").read_text().splitlines()
    test_lines = (tmp_path / "mydb_test.txt").read_text().splitlines()

    assert counts["train"] + counts["valid"] + counts["test"] == 10
    assert len(train_lines) == counts["train"]
    assert len(valid_lines) == counts["valid"]
    assert len(test_lines) == counts["test"]

    # All lines end with _final
    all_lines = train_lines + valid_lines + test_lines
    assert all(line.endswith("_final") for line in all_lines)

    # No duplicates across splits
    assert len(set(all_lines)) == 10

    # With 10% test and 10% valid, eval sets get ~1 entry each; train gets the rest
    assert counts["test"] >= 1
    assert counts["valid"] >= 1
    assert counts["train"] >= 8


def test_split_clusters_ratio_normalization(tmp_path: Path) -> None:
    rows = [(f"p{i:02d}", f"C{i:06d}") for i in range(10)]
    _write_clusters_tsv(tmp_path / "final_clusters.tsv", rows)

    counts_normalized = split_clusters(tmp_path, "a", train=0.8, valid=0.1, test=0.1, seed=1)
    counts_unnormalized = split_clusters(tmp_path, "b", train=8.0, valid=1.0, test=1.0, seed=1)

    assert counts_normalized == counts_unnormalized


def test_split_clusters_large_clusters_go_to_train(tmp_path: Path) -> None:
    # One big cluster (3 members, threshold=3 → qualifies as large), rest are singletons
    rows = [("big0", "C000001"), ("big1", "C000001"), ("big2", "C000001")]
    rows += [(f"s{i}", f"C{i+2:06d}") for i in range(6)]
    _write_clusters_tsv(tmp_path / "final_clusters.tsv", rows)

    counts = split_clusters(tmp_path, "x", train=0.5, valid=0.25, test=0.25, seed=0, max_cluster_size=3)

    train_lines = (tmp_path / "x_train.txt").read_text().splitlines()
    assert "big0_final" in train_lines
    assert "big1_final" in train_lines
    assert "big2_final" in train_lines
    assert counts["train"] + counts["valid"] + counts["test"] == 9

    # Singletons below the threshold: largest-first means test/valid get filled
    # before the remaining singletons go to train
    assert counts["test"] >= 1
    assert counts["valid"] >= 1


def test_split_clusters_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="final_clusters.tsv"):
        split_clusters(tmp_path, "x")


def test_split_clusters_largest_eligible_clusters_go_to_eval(tmp_path: Path) -> None:
    # Two 3-member clusters + six singletons; 3-member clusters should land in
    # test and valid (largest-first), singletons fill train
    rows = [
        ("a0", "C000001"), ("a1", "C000001"), ("a2", "C000001"),
        ("b0", "C000002"), ("b1", "C000002"), ("b2", "C000002"),
    ]
    rows += [(f"s{i}", f"C{i+3:06d}") for i in range(6)]
    _write_clusters_tsv(tmp_path / "final_clusters.tsv", rows)

    # 12 total small entries; test=0.25 → thresh=3, valid=0.25 → thresh=6
    counts = split_clusters(tmp_path, "y", train=0.5, valid=0.25, test=0.25, seed=0)

    test_lines = (tmp_path / "y_test.txt").read_text().splitlines()
    valid_lines = (tmp_path / "y_valid.txt").read_text().splitlines()
    train_lines = (tmp_path / "y_train.txt").read_text().splitlines()

    # The two 3-member clusters fill exactly the test (3) and valid (3) quotas
    assert counts["test"] == 3
    assert counts["valid"] == 3
    assert counts["train"] == 6

    # 3-member clusters are in eval; singletons are in train
    eval_ids = {line.removesuffix("_final") for line in test_lines + valid_lines}
    assert {"a0", "a1", "a2"} <= eval_ids or {"b0", "b1", "b2"} <= eval_ids
    train_ids = {line.removesuffix("_final") for line in train_lines}
    assert all(f"s{i}" in train_ids for i in range(6))


def test_split_clusters_single_cluster_goes_to_test(tmp_path: Path) -> None:
    # With one cluster and test-first ordering, the single cluster fills test
    rows = [("a", "C000001"), ("b", "C000001")]
    _write_clusters_tsv(tmp_path / "final_clusters.tsv", rows)

    counts = split_clusters(tmp_path, "s", train=0.8, valid=0.1, test=0.1, seed=0)

    assert counts["test"] == 2
    assert counts["valid"] == 0
    assert counts["train"] == 0
