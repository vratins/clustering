# clustering

Download structural data from PDB-REDO and cluster local PDB/mmCIF complexes using
chain-level sequence evidence from MMseqs and multimer structural evidence from Foldseek.

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
files through generated symlink directories. Biotite is used to extract every
protein-chain sequence for MMseqs; final clusters are still assigned per PDB entry,
not per chain.

## Run

```bash
uv run pdbcluster run \
  --data-dir /path/to/pdbredo \
  --out-dir /path/to/clusters \
  --seq-id 0.30 \
  --seq-cov 0.80 \
  --tm 0.50 \
  --struct-cov 0.80 \
  --max-seqs 0 \
  --gpu-devices 0,1,2,3 \
  --threads 32
```

The pipeline first runs MMseqs all-vs-all on chain FASTA records. Passing chain
hits are collapsed into structure-level sequence edges, and their connected
components gate Foldseek. Foldseek `easy-multimercluster` runs only inside
non-singleton sequence components, with monomers included via
`--monomer-include-mode 0` and `--min-aligned-chains 1`.

Final fused edges require both sequence support and Foldseek multimer support.
Final clusters are connected components of that fused graph.

`--max-seqs 0` disables tool defaults by resolving to an exhaustive value for each
stage: total chain count for MMseqs and current sequence-component size for
Foldseek. A positive `--max-seqs` is passed directly to both tools and should be
treated as an approximate sensitivity/speed cap.

## Caching

Tool stages write sidecar `*.params.json` files. A cached stage is reused only when
its expected outputs exist and the cached params exactly match the current command,
thresholds, resolved `--max-seqs`, tool version, and input fingerprint. Changing
inputs or relevant arguments reruns the affected stage automatically.

`pdbcluster run` prints concise progress lines as stages start, finish, hit the
cache, skip singleton sequence components, or fail. The same events are appended
to `progress.jsonl` as one JSON object per event. They include component IDs where
relevant and the tool log path for command stages. External tool stdout and stderr
are written to the stage `.log` file along with the command, exit code, and
elapsed time.

## Outputs

- `manifest.tsv`: one row per extracted protein chain, including stable `chain_uid`,
  source structure, native chain ID, sequence length, and parse status.
- `work/chains.fasta`: chain-level FASTA input for MMseqs.
- `work/structures/`: symlinks to native structure files.
- `progress.jsonl`: append-only stage progress events; the CLI also prints these
  events live while running.
- `mmseqs/seq_edges.tsv`: raw chain-level MMseqs evidence.
- `mmseqs/seq_edges.log`: MMseqs command, stdout/stderr, exit code, and elapsed time.
- `mmseqs/structure_seq_edges.tsv`: passing sequence evidence collapsed to PDB-entry
  pairs with best chain support and evidence count.
- `foldseek/components/<Sxxxxxx>/`: per-sequence-component Foldseek work dirs,
  including `structure_cluster.log` for Foldseek stdout/stderr.
- `foldseek/structure_edges.tsv`: multimer structure edges parsed from Foldseek
  `_cluster_report` files. This is sequence-gated, not a global all-vs-all
  structure clustering.
- `final_edges.tsv`: fused structure-level edges with sequence support summary,
  multimer TM/coverage, interface LDDT, and source sequence component.
- `final_clusters.tsv`: final cluster assignment per PDB entry.
- `run_manifest.json`: commands run, versions, config, counts, and elapsed time.

## Download PDB-REDO Data

The downloader accepts the RCSB search filters as command-line flags and writes the
expected directory layout:

```bash
python download_pdb_redo.py /path/to/output \
  --workers 16 \
  --include-pdb \
  --method "X-RAY DIFFRACTION" \
  --max-resolution 3.0 \
  --max-rfree 0.25 \
  --min-residues 50 \
  --max-residues 500 \
  --polymer-entity-type "Protein (only)"
```

To query without downloading, write one `<pdb_id>_final` stem per matching entry
to a user-named file with `--file-list`:

```bash
python download_pdb_redo.py /path/to/output \
  --file-list queried_files.txt \
  --method "X-RAY DIFFRACTION"
```

Run `python download_pdb_redo.py --help` to see the full set of knobs and defaults.
