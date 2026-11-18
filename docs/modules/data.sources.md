---
tags: [module, data]
aliases: [sources.py, t2c data download]
---
# data.sources
**File:** `src/t2c/data/sources.py` · **Libraries:** [[Libraries#huggingface_hub|huggingface_hub]], `tarfile`, `urllib`, `hashlib`
**Concepts:** [[Reproducible environments]] · **Data:** [[Data overview]]

The list of the five upstream sources and the code that downloads them — always at an
**exact commit**, never "latest", so both clusters get identical bytes.

| Name | Where | Revision | Role | Note |
|---|---|---|---|---|
| `t2cq` | HF `ricemonster/IEEE-T-ASE` (a *model* repo) | `4091ce30` | train | [[Text-to-CadQuery dataset]] |
| `cadcoder` | HF dataset `gudo7208/CAD-Coder` | `cd716100` | train | [[CAD-Coder dataset]] |
| `t2cb` | HF dataset `AICAD/Text2CAD-Bench` | `76428d5f` | benchmark | [[Text2CAD-Bench]] |
| `cqe` | HF dataset `drwahl/cadqueryeval` | `7504b0d7` | benchmark | [[CadQueryEval]] |
| `cadprompt` | GitHub `Kamel773/CAD_Code_Generation` | `33dcecd6` | benchmark | [[CADPrompt]] |

Each `Source` also records the file patterns to fetch (only what we use), the licence and
a one-line note.

## How downloading works
- **Hugging Face** — `snapshot_download(repo_id, revision=<commit>, allow_patterns=…)`.
- **GitHub** — fetch the tarball of that commit from `codeload.github.com`, extract only
  the wanted folder with `tarfile`'s `filter="data"` (blocks path tricks inside archives),
  skip hidden files.
- After a download it writes `data/raw/<name>/.t2c_source.json`: repo, revision, licence,
  and SHA-256 hashes of every file for small sources. If that stamp already names the
  pinned revision, the download is skipped (`--force` re-downloads).

`describe()` prints the table with `[x]` for what is present (shown by `t2c data download`).

To move to a newer upstream version: change the revision here on purpose, re-download, and
re-run [[data.prepare]] / [[bench.suites]] — results from the old and new data should not
be mixed silently.

**Used by:** [[cli]] (`t2c data download`). Everything else reads `data/raw/` through [[paths]].
