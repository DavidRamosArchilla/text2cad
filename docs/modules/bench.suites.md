---
tags: [module, benchmark]
aliases: [suites.py, t2c bench prepare]
---
# bench.suites
**File:** `src/t2c/bench/suites.py` · **Libraries:** [[Libraries#pandas and pyarrow|pandas]], [[Libraries#trimesh|trimesh]], [[Libraries#numpy|numpy]], `csv` + [[cad.sandbox]]
**Benchmarks:** [[Benchmarks overview]]

The five benchmark classes. Each knows how to turn its raw download into samples +
ground-truth meshes (`prepare()`, run by `t2c bench prepare`) and, where needed, how its
protocol differs from the default.

| Class | `name` | Prepared from | Ground truth | Protocol differences |
|---|---|---|---|---|
| `Text2CADBench` | `t2cb` | `release.csv` (GB18030-encoded) → 151 parts × {geo, seq} | **none public**; picked up automatically from `data/raw/t2cb_gt/<id>.step` if you get it (STEP → mesh via cadquery) | paper's prompt, 3 retries, headline IR |
| `CadQueryEval` | `cqe` | `tasks.parquet` + reference STLs | STL → `.npz`, alternate references kept | upstream prompt verbatim; the script must write `output.stl` in protocol mode; adds the six pass/fail checks; extracts the published leaderboard to `published.json` |
| `CADPrompt` | `cadprompt` | 200 object folders × 2 prompt variants | `Ground_Truth.stl` (rebuilt from the expert script if the STL had holes; dropped if neither works) | groups `measured` / `abstract` |
| `DeepCADText` | `deepcad` | 500 random rows (seed 20261115) of the verified t2cq **test** split | the reference script executed in the sandbox | — |
| `SynthBench` | `synth` | the generator's **test** split ([[data.synth]]) | the generator's script executed | groups = part family |

What each benchmark *represents* and how to read it: [[Text2CAD-Bench]],
[[CadQueryEval]], [[CADPrompt]], [[DeepCAD held-out]], [[Synth benchmark]].

Details worth knowing:
- `_stl_to_npz` converts reference STLs and measures `open_frac`, so a broken reference is
  noticed at preparation time, not during an evaluation.
- `CADPrompt.uids()` and `DeepCADText.uids()` expose the DeepCAD ids of their objects;
  [[data.build]] uses them to keep those objects out of training.
- CadQueryEval in *native* mode tessellates at 0.1 mm (what `cq.exporters.export` would
  use), so the six checks see the same mesh resolution in both modes.

**Used by:** [[cli]], via [[bench.base]] (registry). **Tested in:** not directly (needs the
downloads); the scoring path it configures is covered by [[test_eval]] and [[test_geometry]].
