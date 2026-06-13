from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MMSEQS_GPU_URL = "https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
FOLDSEEK_GPU_URL = "https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz"


@dataclass(frozen=True)
class ToolPaths:
    mmseqs: Path
    foldseek: Path


def resolve_tool(
    name: str, explicit: Path | None = None, tool_dir: Path = Path(".tools")
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.exists() else None

    local = tool_dir.expanduser().resolve() / name / "bin" / name
    if local.exists():
        return local

    found = shutil.which(name)
    return Path(found).resolve() if found else None


def resolve_tools(
    tool_dir: Path = Path(".tools"),
    mmseqs_path: Path | None = None,
    foldseek_path: Path | None = None,
) -> ToolPaths:
    mmseqs = resolve_tool("mmseqs", mmseqs_path, tool_dir)
    foldseek = resolve_tool("foldseek", foldseek_path, tool_dir)
    missing = [name for name, path in (("mmseqs", mmseqs), ("foldseek", foldseek)) if path is None]
    if missing:
        raise FileNotFoundError(
            "missing required tools: "
            + ", ".join(missing)
            + ". Run `uv run pdbcluster bootstrap-tools` or provide explicit paths."
        )
    return ToolPaths(mmseqs=mmseqs, foldseek=foldseek)


def bootstrap_tools(
    tool_dir: Path = Path(".tools"), force: bool = False
) -> dict[str, dict[str, str]]:
    tool_dir = tool_dir.expanduser().resolve()
    tool_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "mmseqs": _download_and_extract("mmseqs", MMSEQS_GPU_URL, tool_dir, force),
        "foldseek": _download_and_extract("foldseek", FOLDSEEK_GPU_URL, tool_dir, force),
    }
    manifest_path = tool_dir / "tools_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def tool_version(path: Path) -> str:
    for args in ([str(path), "version"], [str(path), "--version"]):
        try:
            result = subprocess.run(args, check=False, capture_output=True, text=True)
        except OSError as exc:
            return f"error: {exc}"
        output = (result.stdout or result.stderr).strip()
        if result.returncode == 0 and output:
            return output.splitlines()[0]
    return "unknown"


def nvidia_smi_summary() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "nvidia-smi not found"
    result = subprocess.run([nvidia_smi, "-L"], check=False, capture_output=True, text=True)
    output = (result.stdout or result.stderr).strip()
    return output if output else f"nvidia-smi exited {result.returncode}"


def build_mmseqs_search_cmd(
    mmseqs: Path,
    fasta_path: Path,
    output_tsv: Path,
    tmp_dir: Path,
    seq_id: float,
    seq_cov: float,
    threads: int,
    gpu: bool,
    max_seqs: int,
) -> list[str]:
    cmd = [
        str(mmseqs),
        "easy-search",
        str(fasta_path),
        str(fasta_path),
        str(output_tsv),
        str(tmp_dir),
        "--format-output",
        "query,target,fident,qcov,tcov,alnlen,evalue,bits",
        "--alignment-mode",
        "3",
        "-a",
        "--min-seq-id",
        str(seq_id),
        "-c",
        str(seq_cov),
        "--cov-mode",
        "0",
        "--max-seqs",
        str(max_seqs),
        "--threads",
        str(threads),
    ]
    if gpu:
        cmd.extend(["--gpu", "1"])
    return cmd


def build_foldseek_multimercluster_cmd(
    foldseek: Path,
    structures_dir: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    tm_threshold: float,
    struct_cov: float,
    threads: int,
    gpu: bool,
    max_seqs: int,
) -> list[str]:
    cmd = [
        str(foldseek),
        "easy-multimercluster",
        str(structures_dir),
        str(cluster_prefix),
        str(tmp_dir),
        "--alignment-type",
        "1",
        "--multimer-tm-threshold",
        str(tm_threshold),
        "-c",
        str(struct_cov),
        "--cov-mode",
        "0",
        "--monomer-include-mode",
        "0",
        "--min-aligned-chains",
        "1",
        "--max-seqs",
        str(max_seqs),
        "--threads",
        str(threads),
        "--remove-tmp-files",
        "0",
    ]
    if gpu:
        cmd.extend(["--gpu", "1"])
    return cmd


def build_foldseek_multimer_report_cmd(
    foldseek: Path,
    query_db: Path,
    result_db: Path,
    report_path: Path,
    threads: int,
) -> list[str]:
    return [
        str(foldseek),
        "createmultimerreport",
        str(query_db),
        str(query_db),
        str(result_db),
        str(report_path),
        "--threads",
        str(threads),
    ]


def run_command(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.write(f"started_at={started:.3f}\n\n")
        log.flush()
        process = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
        elapsed = time.time() - started
        log.write(f"\nexit_code={process.returncode}\n")
        log.write(f"elapsed_seconds={elapsed:.3f}\n")
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log_path}")


def _download_and_extract(
    name: str,
    url: str,
    tool_dir: Path,
    force: bool,
) -> dict[str, str]:
    install_dir = tool_dir / name
    binary = install_dir / "bin" / name
    if binary.exists() and not force:
        return {
            "url": url,
            "path": str(binary),
            "sha256": "",
            "status": "already_present",
            "version": tool_version(binary),
        }

    with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=tool_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        archive_path = tmp_dir / f"{name}.tar.gz"
        sha256 = _download(url, archive_path)
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, extract_dir)

        extracted_root = extract_dir / name
        if not extracted_root.exists():
            roots = [path for path in extract_dir.iterdir() if path.is_dir()]
            if len(roots) != 1:
                raise RuntimeError(f"could not find extracted {name} directory in archive")
            extracted_root = roots[0]

        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(extracted_root), install_dir)

    return {
        "url": url,
        "path": str(binary),
        "sha256": sha256,
        "status": "installed",
        "version": tool_version(binary),
    }


def _download(url: str, path: Path) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if os.path.commonpath([destination, member_path]) != str(destination):
            raise RuntimeError(f"unsafe path in archive: {member.name}")
    archive.extractall(destination)
