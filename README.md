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
  --complex-seq-cov 0.50 \
  --tm 0.50 \
  --struct-cov 0.80 \
  --max-seqs 0 \
  --gpu-devices 0,1,2,3 \
  --threads 32
```

The pipeline first runs MMseqs all-vs-all on chain FASTA records. Passing chain
hits are collapsed into structure-level sequence edges: for each candidate
PDB-entry pair, chains are matched 1-to-1 with an optimal (Hungarian) assignment,
and the pair is kept only if the matched chains cover at least `--complex-seq-cov`
of *both* entries. This `min(qcov, tcov)` complex-coverage rule keeps assemblies of
different stoichiometry (e.g. a monomer vs. its homo-tetramer) in separate clusters.

The connected components of the sequence-edge graph then gate Foldseek: structural
comparison only runs *within* a component, so Foldseek never wastes TMalign on
sequence-dissimilar pairs, and sequence singletons skip Foldseek entirely. Each
component with two or more members is compared all-vs-all with a single
`foldseek easy-multimersearch` call (`--alignment-type 1`), with monomers and
monomer-vs-multimer pairs handled via `--monomer-include-mode 0` and
`--min-aligned-chains 1`. Structure coverage is enforced at alignment time by
Foldseek's `-c <struct-cov>`.

A pair survives fusion only if it has **both** a sequence edge and a structure edge
with `min(qTM, tTM) >= --tm`. Final clusters are the connected components of that
fused graph (single linkage), giving one cluster assignment per PDB entry.

`--max-seqs 0` (the default) means "never truncate": it resolves to the total chain
count for the MMseqs search and to the current sequence-component size for each
Foldseek search. A positive `--max-seqs` is passed directly to both tools as an
approximate sensitivity/speed cap.

Pass `--force` to ignore all cached stage outputs and recompute from scratch. Use
`--no-gpu` to run the searches on CPU.

## Caching

Tool stages write sidecar `*.params.json` files. A cached stage is reused only when
its expected outputs exist and the cached params exactly match the current command,
thresholds, resolved `--max-seqs`, tool version, and input fingerprint. Changing
inputs or relevant arguments reruns the affected stage automatically; Foldseek is
cached per sequence component, so adding new structures only recomputes the
components they touch. Pass `--force` to bypass every cache and recompute.

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
  pairs with matched chains, complex coverage, and evidence count.
- `foldseek/components/<Sxxxxxx>/`: per-sequence-component Foldseek work dirs,
  including the component structure symlinks and `search.log` for Foldseek
  stdout/stderr.
- `foldseek/structure_edges.tsv`: multimer structure edges (`min(qTM, tTM)` per
  PDB-entry pair) parsed from Foldseek `easy-multimersearch` `_report` files. This is
  sequence-gated, not a global all-vs-all structure clustering. Coverage and
  interface LDDT are recorded as `NA` (Foldseek's `-c` enforces coverage during
  alignment; it does not emit per-complex coverage in this report).
- `final_edges.tsv`: fused edges that cleared both sequence and structure thresholds,
  with the sequence-support summary, multimer TM scores, and source sequence component.
- `final_clusters.tsv`: final cluster assignment per PDB entry — columns `pdb_id`,
  `final_cluster`, `final_representative`, `sequence_component`, `sequence_length`.
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

### Query without downloading

Write one `<pdb_id>_final` stem per matching entry to a file. No output directory
is needed:

```bash
python download_pdb_redo.py \
  --file-list queried_files.txt \
  --method "X-RAY DIFFRACTION" \
  --max-resolution 3.0 --max-rfree 0.25 \
  --min-residues 50 --max-residues 500 \
  --polymer-entity-type "Protein (only)"
```

### Check an existing download directory

To see which entries from the query are absent or incomplete in an existing output
directory, use `--check-dir`. This logs a count to stderr and, combined with
`--file-list`, writes only the missing stems to a file:

```bash
python download_pdb_redo.py \
  --check-dir /path/to/output \
  --file-list missing_files.txt \
  [filter flags...]
```

### Download only the missing entries

`--download-missing` (requires `--check-dir`) downloads the entries that are absent
or incomplete. Before starting, it reads `download_manifest.tsv` from the output
directory (if present) and skips any entries that previously failed, so interrupted
runs can be resumed cleanly:

```bash
python download_pdb_redo.py \
  --check-dir /path/to/output \
  --download-missing \
  [filter flags...]
```

Run `python download_pdb_redo.py --help` to see the full set of knobs and defaults.
