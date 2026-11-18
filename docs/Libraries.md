---
tags: [index]
aliases: [Dependencies]
---
# Libraries

Every third-party library the project uses, what it is, where it appears and why it was
chosen. Exact versions are fixed in `uv.lock`; the version ranges are in `pyproject.toml`.
How the environment is built: [[Reproducible environments]] and [[Scripts and clusters]].

The dependencies come in two groups (see `pyproject.toml`):

- **core** — enough for data preparation and scoring on a CPU-only machine.
- **`gpu` extra** — model inference and training (torch, transformers, TRL, vLLM…).
  `scripts/bootstrap.sh --cpu` skips it.

## Summary

| Library | What it is | Used in | For |
|---|---|---|---|
| [[#cadquery]] | Python CAD library on the OpenCascade kernel | [[cad.sandbox]] (worker) | executing scripts, building and meshing solids |
| [[#numpy]] | n-dimensional arrays | [[cad.geometry]], [[cad.metrics]], [[bench.suites]] | all geometry maths |
| [[#scipy]] | scientific algorithms | [[cad.metrics]] | nearest-neighbour search (KD-tree) |
| [[#trimesh]] | triangle-mesh toolkit | [[cad.metrics]], [[cad.sandbox]], [[bench.suites]] | loading STL files, mesh clean-up, watertight / component checks |
| [[#datasets]] | Hugging Face tabular datasets | [[train.sft]], [[train.grpo]] | the input format TRL trainers expect |
| [[#huggingface_hub]] | Hugging Face Hub client | [[data.sources]] (and inside transformers/vLLM) | downloading data and models at pinned revisions |
| [[#pandas and pyarrow]] | dataframes, Parquet | [[bench.suites]], [[bench.cqe]] | reading CadQueryEval's Parquet files |
| [[#PyYAML]] | YAML parser | configs everywhere | reading `configs/**/*.yaml` |
| [[#tqdm]] | progress bars | [[cad.sandbox]] | progress of long verification batches |
| [[#openai]] | OpenAI API client | [[infer.backends]] | evaluating API models |
| [[#torch]] | deep-learning framework | training, inference | tensors on GPU, autocast |
| [[#transformers]] | model and tokenizer library | [[infer.backends]], [[train.sft]], [[train.grpo]], [[train.callbacks]] | loading Qwen3, chat templates, `generate`, the Trainer |
| [[#TRL]] | training recipes on top of transformers | [[train.sft]], [[train.grpo]] | `SFTTrainer`, `GRPOTrainer` |
| [[#accelerate]] | device / distributed launcher | under the Trainer | multi-GPU runs |
| [[#vLLM]] | fast LLM inference engine | [[infer.backends]], GRPO rollouts | batched generation |
| [[#tensorboard]] | training-curve viewer | `report_to` in configs | loss, reward and benchmark curves |
| [[#pytest]] | test runner | `tests/` | the test suite |
| [[#ruff]] | linter / formatter | development | code style |
| [[#jinja2]] | template engine | inside transformers | rendering chat templates |

Tools that are not Python imports: **uv** (environment manager, see
[[Reproducible environments]]), **hatchling** (build backend that makes `t2c` an installable
package with a `t2c` command), **Slurm** ([[Slurm]]).

---

## cadquery
Python library for parametric CAD. A script builds a solid step by step
(`cq.Workplane("XY").box(10, 20, 5).faces(">Z").workplane().hole(3)`); underneath, the
**OCP** package (`cadquery-ocp`) gives Python access to **OpenCascade (OCCT)**, the C++
geometry kernel that does the real work. The models we train *write* CadQuery; the
verifier *runs* it.

- Used in: `src/t2c/cad/worker.py` (`exec` of the script, `Shape.Volume()`, `isValid()`,
  `tessellate()`), `bench/suites.py` (`importers.importStep` for STEP ground truth).
- Why: it is the target language of every benchmark here; scripts are plain Python, so
  LLMs already know the syntax.
- Note: OCCT can crash the whole process (segfault) on bad geometry — the reason each
  script runs in its own process ([[cad.sandbox]]).
- Docs: <https://cadquery.readthedocs.io/en/latest/> · concepts:
  <https://cadquery.readthedocs.io/en/latest/primer.html> · source:
  <https://github.com/CadQuery/cadquery>, <https://github.com/CadQuery/OCP>. Background: [[CadQuery and B-rep]].

## numpy
Array maths. Meshes are numpy arrays (`v`: N×3 vertex coordinates, `f`: M×3 triangle
indices); the voxeliser, surface sampling, volume and alignment are vectorised numpy.
Docs: <https://numpy.org/doc/stable/>

## scipy
Only `scipy.spatial.cKDTree`: a KD-tree that finds the nearest point in a cloud quickly.
Needed for Chamfer distance, Hausdorff distance and ICP alignment in [[cad.metrics]].
Docs: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html> ·
background: [[Comparing 3D shapes]].

## trimesh
Loads STL files (benchmark references, the `output.stl` CadQueryEval scripts write), merges
duplicate vertices, removes degenerate triangles, tests watertightness, counts connected
components and computes mesh volume. Used in [[cad.metrics]] (`cqe_checks`), the worker
(CadQueryEval protocol) and [[bench.suites]] (STL → `.npz`).
Docs: <https://trimesh.org/>

> [!note] Deliberately not used: open3d
> The official CadQueryEval scorer uses open3d for point clouds and alignment. Its current
> wheels need glibc ≥ 2.35 but this cluster runs Rocky Linux 9 (glibc 2.34), and its
> RANSAC alignment is random. [[cad.metrics]] re-implements the same checks with numpy +
> scipy + trimesh; [[bench.cqe]] shows the verdicts agree.

## datasets
Hugging Face's dataset library. Used only for `Dataset.from_list(...)`, because TRL's
trainers take a `datasets.Dataset`. Our data files are plain JSONL.
Docs: <https://huggingface.co/docs/datasets/index>

## huggingface_hub
`snapshot_download(repo_id, revision=<commit>)` in [[data.sources]] downloads exactly one
commit of a dataset repo. transformers and vLLM use the same library to download model
weights into the cache (`HF_HOME`). Docs:
<https://huggingface.co/docs/huggingface_hub/guides/download>

## pandas and pyarrow
CadQueryEval ships its tasks and its leaderboard attempts as Parquet files;
`pandas.read_parquet` (which uses pyarrow) reads them in [[bench.suites]] and
[[bench.cqe]]. Docs: <https://pandas.pydata.org/docs/>, <https://arrow.apache.org/docs/python/>

## PyYAML
Reads every config: model specs, training configs, data recipes, published references.
Also parses `key=value` command-line overrides (`yaml.safe_load("1e-5")` → a float).
Docs: <https://pyyaml.org/wiki/PyYAMLDocumentation>

## tqdm
Progress bars for long verification batches (`progress=` argument of `CadExecutor.run`).
Docs: <https://tqdm.github.io/>

## openai
The official client for OpenAI-compatible chat APIs. `OpenAIBackend` in [[infer.backends]]
uses it for frontier models, OpenRouter, or a remote vLLM server.
Docs: <https://github.com/openai/openai-python>, <https://openrouter.ai/docs>

## torch
PyTorch: tensors on the GPU, automatic differentiation, `torch.autocast` for bf16
compute. Used directly only in small places (dtype choice, `no_grad`, autocast, cache
clearing); the heavy use is inside transformers / TRL / vLLM.
Docs: <https://pytorch.org/docs/stable/index.html> · [[Mixed precision]]

## transformers
Hugging Face's model library.
- `AutoTokenizer` and `apply_chat_template` → the exact prompt text ([[Tokenizers and chat templates]]).
- `AutoModelForCausalLM` → loads Qwen3 weights for training and for the `hf` backend.
- `model.generate` → text generation in `hf_generate` ([[infer.backends]], [[train.callbacks]]).
- `TrainerCallback` → base class of the in-training benchmark callback.
- `Trainer` / `TrainingArguments` → the training loop TRL builds on (optimizer, schedule,
  checkpoints, logging).

Docs: <https://huggingface.co/docs/transformers/index> ·
callbacks: <https://huggingface.co/docs/transformers/main_classes/callback>

## TRL
"Transformer Reinforcement Learning", Hugging Face's post-training library.
- `SFTTrainer` + `SFTConfig` in [[train.sft]].
- `GRPOTrainer` + `GRPOConfig` in [[train.grpo]]: samples groups of answers (with vLLM),
  calls our reward function, computes advantages and the policy loss.

Everything under `train:` / `grpo:` in our training configs is passed straight to
`SFTConfig` / `GRPOConfig`. Docs: <https://huggingface.co/docs/trl/index> ·
<https://huggingface.co/docs/trl/sft_trainer> · <https://huggingface.co/docs/trl/grpo_trainer>

## accelerate
Runs the same training script on 1 or many GPUs. The Trainer uses it internally; you call
it explicitly for multi-GPU (`accelerate launch --num_processes 2 -m t2c.cli train sft …`).
Docs: <https://huggingface.co/docs/accelerate/index> · [[Multi-GPU training]]

## vLLM
High-throughput inference engine. Two uses:
1. `VLLMBackend` in [[infer.backends]] — the default way to evaluate local models.
2. Inside `GRPOTrainer` (`use_vllm: true`, `vllm_mode: colocate`) — generating the RL rollouts.

Docs: <https://docs.vllm.ai/en/latest/> · background: [[vLLM]]

## tensorboard
Training curves. `report_to: [tensorboard]` in a training config writes event files under
`outputs/train/<run>/`; open them with `tensorboard --logdir outputs/train`.
Docs: <https://www.tensorflow.org/tensorboard/get_started> · [[Experiment tracking]]

## pytest
Runs `tests/`. Fixtures (shared set-up objects such as the CAD `executor`) are the main
feature used. Docs: <https://docs.pytest.org/en/stable/getting-started.html>,
<https://docs.pytest.org/en/stable/how-to/fixtures.html> · [[Tests overview]]

## ruff
Linter and formatter (`ruff check src tests`). Rules are in `pyproject.toml`.
Docs: <https://docs.astral.sh/ruff/>

## jinja2
Chat templates are small Jinja programs stored with each tokenizer; transformers renders
them with jinja2. It is not imported by our code. Docs: <https://jinja.palletsprojects.com/>

## Python standard library highlights
Less obvious standard-library modules doing important work:

| Module | Where | For |
|---|---|---|
| `ast` | [[cad.codeutil]] | static safety check of scripts; renaming a variable by syntax-tree position |
| `os.fork`, `selectors`, `signal`, `pickle`, `struct` | [[cad.sandbox]] (zygote) | forking workers, waiting on many pipes, killing, message framing |
| `resource` | [[cad.sandbox]] (worker) | memory / CPU / file-size limits per script |
| `ctypes` | [[cad.sandbox]] (zygote) | calling the Linux `unshare` system call for network isolation |
| `subprocess`, `threading`, `queue` | [[cad.sandbox]] | starting the zygote, draining results without deadlock |
| `argparse` | [[cli]] | the `t2c` command line |
| `tarfile`, `urllib` | [[data.sources]] | downloading a GitHub repository at one commit |
| `hashlib` | several | de-duplication, content hashes of benchmarks and datasets |
| `random` | [[data.synth]], [[data.build]] | seeded, reproducible sampling |
| `gzip`, `json` | [[results.store]] | the results registry |
| `csv` | [[bench.suites]] | Text2CAD-Bench's CSV (GB18030-encoded) |
