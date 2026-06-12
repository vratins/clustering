from pathlib import Path

from pdbcluster.discovery import discover_entries, normalize_item_id


def test_discover_entries_prefers_cif(tmp_path: Path) -> None:
    entry_dir = tmp_path / "1abc"
    entry_dir.mkdir()
    (entry_dir / "1abc_final.cif").write_text("cif")
    (entry_dir / "1abc_final.pdb").write_text("pdb")

    entries = discover_entries(tmp_path)

    assert len(entries) == 1
    assert entries[0].pdb_id == "1abc"
    assert entries[0].path.name == "1abc_final.cif"
    assert entries[0].format == "cif"


def test_discover_entries_uses_pdb_fallback(tmp_path: Path) -> None:
    entry_dir = tmp_path / "2def"
    entry_dir.mkdir()
    (entry_dir / "2def_final.pdb").write_text("pdb")

    entries = discover_entries(tmp_path)

    assert len(entries) == 1
    assert entries[0].path.name == "2def_final.pdb"


def test_normalize_item_id_removes_paths_and_structure_suffixes() -> None:
    assert normalize_item_id("/tmp/work/1abc.cif") == "1abc"
    assert normalize_item_id("2def.pdb extra") == "2def"
