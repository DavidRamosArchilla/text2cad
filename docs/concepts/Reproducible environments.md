---
tags: [concept]
aliases: [uv, lockfile, reproducibility]
---
# Reproducible environments

The project must give the same results on two clusters. Four things have to be pinned:

| What | How | Where |
|---|---|---|
| Python packages | `uv.lock` records the exact version (and hash) of every package; `uv sync --frozen` installs exactly that | `scripts/bootstrap.sh` |
| the Python interpreter | uv downloads its own standalone CPython 3.12 (`python-preference = "only-managed"`) instead of using conda or the system Python | `pyproject.toml`, `.python-version` |
| data | every download names an exact commit of its repository | [[data.sources]] |
| randomness | every random choice has a seed; dataset files are byte-identical across runs | [[data.synth]], [[data.build]] |

Checks that it worked: `data/bench/*/info.json` stores a hash of the benchmark samples,
and every result records it together with the git commit and host ([[results.store]]).

**Not pinned yet:** model weights are downloaded by name (`Qwen/Qwen3-1.7B`), i.e. the
latest revision on the Hub. Upstream rarely changes released weights, but for strict
reproducibility a `revision` could be added to the model configs.

GPU numerics are not bit-identical across GPU types or library versions; expect tiny
differences in training, none in data preparation or scoring.

## Learn more
- **uv documentation**: <https://docs.astral.sh/uv/> · Locking and syncing
  <https://docs.astral.sh/uv/concepts/projects/sync/> · Python versions
  <https://docs.astral.sh/uv/concepts/python-versions/>
- **Hugging Face Hub — downloading files** (revisions, cache): <https://huggingface.co/docs/huggingface_hub/guides/download>
