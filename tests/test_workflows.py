import json
from pathlib import Path

from pdbcluster.fusion import FusionThresholds
from pdbcluster.prepare import PreparedChain, PreparedEntry, PreparedInputs
from pdbcluster.tools import ToolPaths
from pdbcluster.workflows import ProgressTracker, _cache_valid, _run_tool_workflows


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


def test_foldseek_runs_only_inside_sequence_components(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    foldseek_calls: list[list[str]] = []

    monkeypatch.setattr("pdbcluster.workflows.tool_version", lambda _path: "tool-v1")

    def fake_run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ran\n")
        if cmd[1] == "easy-search":
            Path(cmd[4]).write_text(
                "a__chain0001\tb__chain0001\t0.9\t1.0\t1.0\t4\t1e-9\t50\n"
                "c__chain0001\td__chain0001\t0.9\t1.0\t1.0\t4\t1e-9\t50\n"
            )
            return

        assert cmd[1] == "easy-multimercluster"
        structures = sorted(path.stem for path in Path(cmd[2]).iterdir())
        foldseek_calls.append(structures)
        prefix = Path(cmd[3])
        a, b = structures
        prefix.with_name(prefix.name + "_cluster_report").write_text(
            f"{a}\t{b}\t1.0\t1.0\t0.8\t0.8\t0.7\t1,0,0,0,1,0,0,0,1\t0,0,0\n"
        )
        prefix.with_name(prefix.name + "_cluster.tsv").write_text(f"{a}\t{a}\n{a}\t{b}\n")

    monkeypatch.setattr("pdbcluster.workflows.run_command", fake_run_command)

    command_log: list[list[str]] = []
    progress = ProgressTracker(tmp_path / "progress.jsonl")
    seq_components, _structure_clusters = _run_tool_workflows(
        prepared,
        tmp_path,
        ToolPaths(mmseqs=Path("mmseqs"), foldseek=Path("foldseek")),
        FusionThresholds(seq_id=0.3, seq_cov=0.8, tm=0.5, struct_cov=0.8),
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
        "foldseek_multimercluster",
        "structure_edge_merge",
    }
    assert any(event["status"] == "done" for event in events)
    assert any("log_path" in event for event in events)

    assert foldseek_calls == [["a", "b"], ["c", "d"]]
    assert seq_components["a"] == seq_components["b"]
    assert seq_components["c"] == seq_components["d"]
    assert seq_components["a"] != seq_components["c"]
    assert command_log[0][command_log[0].index("--max-seqs") + 1] == "4"
    assert [cmd[cmd.index("--max-seqs") + 1] for cmd in command_log[1:]] == ["2", "2"]


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
