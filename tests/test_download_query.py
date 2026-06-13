from pathlib import Path

import pytest

from download_pdb_redo import build_query, write_file_stems


def _params_by_attribute(query: dict) -> dict[str, dict]:
    nodes = query["query"]["nodes"]
    return {node["parameters"]["attribute"]: node["parameters"] for node in nodes}


def test_build_query_uses_search_parameters() -> None:
    query = build_query(
        method="ELECTRON MICROSCOPY",
        max_resolution=4.2,
        max_rfree=0.31,
        min_residues=25,
        max_residues=900,
        polymer_entity_type="Protein (only)",
    )

    params = _params_by_attribute(query)
    assert query["return_type"] == "entry"
    assert params["exptl.method"]["value"] == "ELECTRON MICROSCOPY"
    assert params["rcsb_entry_info.resolution_combined"]["value"] == 4.2
    assert params["refine.ls_R_factor_R_free"]["value"] == 0.31
    assert params["rcsb_entry_info.deposited_polymer_monomer_count"]["value"] == {
        "from": 25,
        "to": 900,
        "include_lower": True,
        "include_upper": True,
    }
    assert params["rcsb_entry_info.selected_polymer_entity_types"]["value"] == "Protein (only)"


def test_parse_args_exposes_search_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    from download_pdb_redo import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "download_pdb_redo.py",
            "/tmp/out",
            "--method",
            "ELECTRON MICROSCOPY",
            "--max-resolution",
            "4.2",
            "--max-rfree",
            "0.31",
            "--min-residues",
            "25",
            "--max-residues",
            "900",
            "--polymer-entity-type",
            "Protein (only)",
            "--file-list",
            "/tmp/file_stems.txt",
        ],
    )

    args = parse_args()

    assert args.method == "ELECTRON MICROSCOPY"
    assert args.max_resolution == 4.2
    assert args.max_rfree == 0.31
    assert args.min_residues == 25
    assert args.max_residues == 900
    assert args.polymer_entity_type == "Protein (only)"
    assert args.file_list == Path("/tmp/file_stems.txt")


def test_write_file_stems_writes_pdbredo_stems(tmp_path: Path) -> None:
    out = tmp_path / "queried_files.txt"

    write_file_stems(["4LSW", "1abc"], out)

    assert out.read_text() == "4lsw_final\n1abc_final\n"


def test_file_list_mode_skips_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from download_pdb_redo import main

    base_dir = tmp_path / "out"
    file_list = tmp_path / "stems.txt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_pdb_redo.py",
            str(base_dir),
            "--file-list",
            str(file_list),
        ],
    )
    monkeypatch.setattr("download_pdb_redo.fetch_all_ids", lambda query, page_size: ["4LSW"])

    def fail_download(*args, **kwargs):
        raise AssertionError("download_one_entry should not be called")

    monkeypatch.setattr("download_pdb_redo.download_one_entry", fail_download)

    assert main() == 0
    assert (base_dir / "ids.txt").read_text() == "4lsw\n"
    assert file_list.read_text() == "4lsw_final\n"
    assert not (base_dir / "download_manifest.tsv").exists()
