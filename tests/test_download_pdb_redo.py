import download_pdb_redo as downloader


def test_build_query_uses_cli_filters() -> None:
    query = downloader.build_query(
        method="X-RAY DIFFRACTION",
        max_resolution=2.5,
        max_rfree=0.2,
        min_residues=40,
        max_residues=120,
        polymer_entity_type="Protein (only)",
    )

    nodes = query["query"]["nodes"]
    assert nodes[0]["parameters"]["value"] == "X-RAY DIFFRACTION"
    assert nodes[1]["parameters"]["value"] == 2.5
    assert nodes[2]["parameters"]["value"] == 0.2
    assert nodes[3]["parameters"]["value"] == {
        "from": 40,
        "to": 120,
        "include_lower": True,
        "include_upper": True,
    }
    assert nodes[4]["parameters"]["value"] == "Protein (only)"
