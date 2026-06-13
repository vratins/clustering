from pathlib import Path

from pdbcluster.discovery import Entry
from pdbcluster.prepare import prepare_inputs


def test_prepare_inputs_reuses_valid_cache(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "data" / "a" / "a_final.cif"
    source.parent.mkdir(parents=True)
    source.write_text("structure a\n")
    entries = [Entry("a", source.resolve(), "cif")]
    calls = 0

    def fake_extract(path: Path) -> list[tuple[str, str]]:
        nonlocal calls
        calls += 1
        return [("A", "ACDE")]

    monkeypatch.setattr("pdbcluster.prepare.extract_chain_sequences", fake_extract)

    first = prepare_inputs(entries, tmp_path / "out")
    second = prepare_inputs(entries, tmp_path / "out")

    assert calls == 1
    assert not first.cached
    assert second.cached
    assert second.chains[0].chain_uid == "a__chain0001"
    assert second.chains[0].sequence == "ACDE"


def test_prepare_inputs_bootstraps_cache_from_existing_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "data" / "a" / "a_final.cif"
    source.parent.mkdir(parents=True)
    source.write_text("structure a\n")
    entries = [Entry("a", source.resolve(), "cif")]

    monkeypatch.setattr(
        "pdbcluster.prepare.extract_chain_sequences", lambda _path: [("A", "ACDE")]
    )
    prepare_inputs(entries, tmp_path / "out")
    params_path = tmp_path / "out" / "work" / "prepare.params.json"
    params_path.unlink()

    def fail_extract(path: Path) -> list[tuple[str, str]]:
        raise AssertionError("cache should avoid sequence extraction")

    monkeypatch.setattr("pdbcluster.prepare.extract_chain_sequences", fail_extract)

    cached = prepare_inputs(entries, tmp_path / "out")

    assert cached.cached
    assert params_path.exists()
    assert cached.entries[0].structure_path.exists()
