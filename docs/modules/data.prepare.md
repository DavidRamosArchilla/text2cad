---
tags: [module, data]
aliases: [prepare.py, t2c data prepare]
---
# data.prepare
**File:** `src/t2c/data/prepare.py` · **Libraries:** `json`, `re`, `hashlib` + [[cad.sandbox]]
**Data:** [[Text-to-CadQuery dataset]] · [[CAD-Coder dataset]]

Turns the two downloaded training corpora into clean, **execution-verified** datasets.
A training pair is only useful if its script really builds a valid part — otherwise the
model learns to write broken code.

## Steps for every (description, script) pair
1. **Read** with a per-source iterator:
   - `t2cq`: JSONL `{input, output}`; the DeepCAD id (`uid`) is taken from the export path
     inside the script (`…/00102156.stl`).
   - `cadcoder`: JSON chat records; the boiler-plate "Please based on the following
     description, create a CAD-Query Code…" is stripped from the prompt; the uid comes from
     `model_path`; the final solid is conventionally called `r`.
2. **Normalise** the script to the `result` contract ([[cad.codeutil]]
   `normalize_reference_script`). Not possible → dropped (`unnormalizable`).
3. **Filter**: scripts longer than 6 000 characters, empty descriptions, exact duplicates
   (same text and code).
4. **Execute** all of them in the sandbox (30 s timeout) with a watertightness check.
   Kept only if the result is a valid solid and the surface is closed
   (`open_frac ≤ 0.005`).
5. **Write** `data/processed/<source>/<split>.jsonl` with rows
   `{id, uid, text, code, volume, bbox, n_solids, n_tris}`, plus `stats.json` with how many
   samples were dropped and why.

## Result (on this cluster)
| Source | Raw pairs | Kept | Main losses |
|---|---|---|---|
| `t2cq` | 110 227 | train 98 473 · val 5 473 · test 5 472 | 587 runtime errors, 165 not watertight |
| `cadcoder` | 83 528 (high + middle + val) | high 8 116 · middle 57 460 · val 8 635 | 8 117 duplicates, 941 unnormalisable |

About 25 minutes on 96 cores (`scripts/submit.sh --cpus 96 -- t2c data prepare`).

`read_split(source, split)` reads a processed file back; used by [[data.build]] and
[[bench.suites]] (`deepcad`, `synth`).

**Used by:** [[cli]]. **Tested in:** indirectly ([[test_build]] writes processed files in the same format; [[test_sandbox]] covers normalisation).
