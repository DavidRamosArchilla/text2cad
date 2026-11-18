---
tags: [module]
aliases: [cli.py, t2c command]
---
# cli
**File:** `src/t2c/cli.py` · **Library:** `argparse` (standard library)

The `t2c` command. `pyproject.toml` declares `t2c = "t2c.cli:main"`, so installing the
package creates the executable `.venv/bin/t2c`. Each sub-command is a small function
that imports the module doing the real work *inside* the function, so the command starts
fast and the light commands work without torch/vLLM.

| Command | Calls | What it does |
|---|---|---|
| `t2c doctor` | `cmd_doctor` | prints Python, paths, free disk, runs one test solid through the sandbox, reports network isolation, GPUs and library versions |
| `t2c data download [names]` | [[data.sources]] | download pinned sources into `data/raw/` |
| `t2c data prepare [sources]` | [[data.prepare]] | verify the training corpora (`--limit` for a quick test) |
| `t2c data synth` | [[data.synth]] | generate the procedural dataset (`--train/--val/--test/--seed`) |
| `t2c data build <recipe>` | [[data.build]] | build an SFT or RL dataset |
| `t2c bench prepare [names]` | [[bench.suites]] | build samples and ground-truth meshes |
| `t2c bench list` | [[bench.base]] | which benchmarks are prepared |
| `t2c bench validate-cqe` | [[bench.cqe]] | compare our CadQueryEval verdicts with the official ones |
| `t2c eval --model M [--bench …]` | [[evalrun.run]] | evaluate a model and record the results |
| `t2c report` | [[results.report]] | rebuild the report |
| `t2c runs` | [[results.store]] | list recorded evaluation runs |
| `t2c train sft <config>`, `t2c train grpo <config>` (+ `key=value` overrides) | [[train.sft]], [[train.grpo]] | train |

Useful `t2c eval` options: `--bench all` (default: every prepared benchmark),
`--prompt-style auto|native|protocol`, `--retries N` (default: the benchmark's protocol),
`--limit N` (smoke test, not recorded in the report), `--backend vllm|hf|openai`,
`--set key=value` (override a model-spec field), `--name`, `--tag`, `--note`.

Tested indirectly: the functions it calls are tested; the argument parsing itself is not.
