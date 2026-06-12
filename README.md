# clustering

Download structural data from PDB-REDO and cluster local PDB/mmCIF files using
sequence evidence from MMseqs and structural evidence from Foldseek.

## Install

Python dependencies are managed with `uv`:

```bash
uv sync --dev
```

MMseqs and Foldseek are external binaries. To install the official project-local
GPU-capable builds:

```bash
uv run pdbcluster bootstrap-tools --tool-dir .tools
uv run pdbcluster validate-tools --tool-dir .tools
```

The bootstrap command downloads:

- `https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz`
- `https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz`

## Input Layout

The clustering CLI expects:

```text
<data_dir>/<pdb_id>/<pdb_id>_final.cif
<data_dir>/<pdb_id>/<pdb_id>_final.pdb
```

CIF is preferred when both CIF and PDB exist. Foldseek receives native structure
files through a generated symlink directory. Biotite is used only to extract the
representative protein sequence needed by MMseqs.

## Run

```bash
uv run pdbcluster run \
  --data-dir /path/to/pdbredo \
  --out-dir /path/to/clusters \
  --seq-id 0.30 \
  --seq-cov 0.80 \
  --tm 0.50 \
  --struct-cov 0.80 \
  --gpu-devices 0,1,2,3 \
  --threads 32
```

The final fused graph keeps an edge only when sequence identity/coverage and
Foldseek TM-score/coverage pass their thresholds. Final clusters are connected
components of that fused graph.

## Outputs

- `manifest.tsv`: selected input files, representative chain, sequence length,
  parse status.
- `work/sequences.fasta`: FASTA input for MMseqs.
- `work/structures/`: symlinks to native structure files for Foldseek.
- `mmseqs/seq_edges.tsv`: all-vs-all sequence edge table.
- `foldseek/structure_edges.tsv`: all-vs-all structural edge table.
- `final_edges.tsv`: edges passing both sequence and structure thresholds.
- `final_clusters.tsv`: final cluster assignment per PDB entry.
- `run_manifest.json`: commands, versions, config, and elapsed time.

## Download PDB-REDO Data

The existing downloader can prepare the expected directory layout:

```bash
python download_pdb_redo.py /path/to/output --workers 16 --include-pdb
```
