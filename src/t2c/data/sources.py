"""Upstream data sources, pinned to exact revisions so every cluster gets the same bytes.

``t2c data download [name ...]`` fetches them into ``$T2C_DATA_DIR/raw/<name>/``.
Bump a revision deliberately (and re-run the preparation) -- never track ``main``.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths


@dataclass(frozen=True)
class Source:
    name: str
    kind: str  # "hf" | "github"
    repo: str
    revision: str
    repo_type: str = "dataset"
    patterns: tuple[str, ...] = ()
    license: str = "unspecified"
    role: str = "train"  # "train" | "benchmark"
    note: str = ""
    extras: dict = field(default_factory=dict)


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            name="t2cq",
            kind="hf",
            repo="ricemonster/IEEE-T-ASE",
            repo_type="model",
            revision="4091ce30ad85b774a9231484f9b40372dc0451d3",
            patterns=("data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl", "README.md"),
            license="unspecified (derived from Text2CAD, CC BY-NC-SA 4.0)",
            note="Text-to-CadQuery (arXiv 2505.06507): ~110k description -> CadQuery pairs, code written by Gemini 2.0 Flash from DeepCAD/Text2CAD.",
        ),
        Source(
            name="cadcoder",
            kind="hf",
            repo="gudo7208/CAD-Coder",
            revision="cd7161002e55a4efaed1bee59f181ac3efc75a2c",
            patterns=(
                "cad_data_train_high.json",
                "cad_data_train_middle.json",
                "cad_data_validation.json",
                "cad_data_test_cot.json",
                "README.md",
                "LICENSE",
            ),
            license="Apache-2.0",
            note="CAD-Coder (NeurIPS 2025): expert-level Text2CAD descriptions -> CadQuery. 'high' = 8k verified, 'middle' = 66k.",
        ),
        Source(
            name="t2cb",
            kind="hf",
            repo="AICAD/Text2CAD-Bench",
            revision="76428d5fe6efc3daa4ed93d4e8a3e5b8e32939c0",
            patterns=("release.csv", "README.md"),
            license="CC BY 4.0",
            role="benchmark",
            note="Public 30% prompt sample of Text2CAD-Bench (151 parts x 2 prompt styles). Ground truth is NOT public.",
        ),
        Source(
            name="cqe",
            kind="hf",
            repo="drwahl/cadqueryeval",
            revision="7504b0d7f74937822a7811221ec613f9f47a3ee9",
            patterns=("data/tasks.parquet", "data/samples.parquet", "reference/*.stl", "README.md"),
            license="mixed (code MIT); evaluation only -- carries a do-not-train canary",
            role="benchmark",
            note="CadQueryEval: 25 parts with reference STLs and every leaderboard attempt of frontier models.",
        ),
        Source(
            name="cadprompt",
            kind="github",
            repo="Kamel773/CAD_Code_Generation",
            revision="33dcecd6087ff7b8a4454b1ba4d56504a6364399",
            patterns=("CADPrompt/",),
            license="unspecified (research use; ICLR 2025 paper artifact)",
            role="benchmark",
            note="CADPrompt: 200 DeepCAD objects with human-written prompts, expert CadQuery code and ground-truth meshes.",
        ),
    ]
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_hf(src: Source, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=src.repo,
        repo_type=src.repo_type,
        revision=src.revision,
        allow_patterns=list(src.patterns) or None,
        local_dir=dest,
    )


def _download_github(src: Source, dest: Path) -> None:
    url = f"https://codeload.github.com/{src.repo}/tar.gz/{src.revision}"
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 - fixed https host
        blob = resp.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel or (src.patterns and not any(rel.startswith(p) for p in src.patterns)):
                continue
            if Path(rel).name.startswith("."):
                continue  # .DS_Store and friends
            m.name = rel
            members.append(m)
        tar.extractall(dest, members=members, filter="data")  # "data" filter blocks path traversal / links


def download(name: str, force: bool = False) -> Path:
    src = SOURCES[name]
    dest = paths.ensure(paths.raw_dir(name))
    stamp = dest / ".t2c_source.json"
    if stamp.exists() and not force:
        have = json.loads(stamp.read_text())
        if have.get("revision") == src.revision:
            return dest
    if src.kind == "hf":
        _download_hf(src, dest)
    elif src.kind == "github":
        _download_github(src, dest)
    else:
        raise ValueError(src.kind)
    files = sorted(p for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != stamp.name)
    manifest = {
        "name": src.name,
        "repo": src.repo,
        "revision": src.revision,
        "license": src.license,
        "n_files": len(files),
        # Hash everything for small sources; for big ones the pinned revision is the guarantee.
        "sha256": {str(p.relative_to(dest)): _sha256(p) for p in files} if len(files) <= 64 else None,
    }
    stamp.write_text(json.dumps(manifest, indent=1))
    return dest


def describe() -> str:
    lines = []
    for s in SOURCES.values():
        have = (paths.raw_dir(s.name) / ".t2c_source.json").exists()
        lines.append(f"{'[x]' if have else '[ ]'} {s.name:<10} {s.role:<9} {s.repo}@{s.revision[:8]}  license: {s.license}\n      {s.note}")
    return "\n".join(lines)
