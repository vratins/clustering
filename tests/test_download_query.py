import pytest

from download_pdb_redo import build_query


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
        ],
    )

    args = parse_args()

    assert args.method == "ELECTRON MICROSCOPY"
    assert args.max_resolution == 4.2
    assert args.max_rfree == 0.31
    assert args.min_residues == 25
    assert args.max_residues == 900
    assert args.polymer_entity_type == "Protein (only)"
