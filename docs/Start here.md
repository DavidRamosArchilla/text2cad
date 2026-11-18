---
tags: [index]
aliases: [Home, Index, README]
---
# Start here

This vault explains the `t2c` repository: what every module does, which libraries it uses
and why, what the tests check, which data goes in and what each benchmark measures. The
theory (GRPO, vLLM, mixed precision, …) is summarised in a few lines and then linked to
good external material — those notes live in `concepts/`.

Open the `docs/` folder as a vault in Obsidian. Words in `[[double brackets]]` are links;
the graph view (`Ctrl+G`) shows how the notes connect. File paths such as
`src/t2c/cad/metrics.py` are relative to the repository root.

> [!info] Snapshot
> Written for commit `4dac18b` (2026-11-16). Each module note names its source file, so
> when the code changes you know which note to update.

## The project in one paragraph

Teach small open LLMs ([[Qwen3]]) to turn a text description of a mechanical part into a
[[CadQuery and B-rep|CadQuery]] Python script that builds that part. Training has two
stages: [[Supervised fine-tuning]] on description → script pairs, then [[GRPO]], a
reinforcement-learning method where the reward comes from *running* the script and
comparing the solid it builds with the correct one. Every model — untouched base models,
your checkpoints, API models — is measured on the same five benchmarks with the same code,
and every result is stored so you can compare runs.

## Suggested reading order

1. [[Pipeline]] — the whole loop on one page, with the command for each step.
2. [[Walkthrough - scoring one answer]] — follow one model answer through the code. It
   touches most modules, so after it the module notes are easy.
3. [[Data overview]] and [[Benchmarks overview]] — what goes in, how it is measured.
4. [[Walkthrough - one GRPO step]] — the reinforcement-learning stage.
5. [[Reading the report]] — how to read `results/REPORT.md`.
6. [[t2c package]] — index of the code; open a module note when you open its file.
7. [[Tests overview]].

If the ML side is new, read these concept notes first, in this order (each has 2–4
external links; pick one): [[LLMs and transformers]] → [[Tokenizers and chat templates]] →
[[Supervised fine-tuning]] → [[Reinforcement learning for LLMs]] → [[GRPO]] → [[vLLM]].

## Map of the vault

**Big picture** — [[Pipeline]] · [[Glossary]] · [[Libraries]]

**Guides** — [[Walkthrough - scoring one answer]] · [[Walkthrough - one GRPO step]] ·
[[Reading the report]] · [[Configs]] · [[Scripts and clusters]]

**Code** (`src/t2c/`) — [[t2c package]]
- top level: [[cli]] · [[paths]] · [[prompts]]
- `cad/` the verifier: [[cad.sandbox]] · [[cad.codeutil]] · [[cad.geometry]] · [[cad.metrics]] · [[cad.reward]]
- `data/`: [[data.sources]] · [[data.prepare]] · [[data.synth]] · [[data.build]]
- `bench/`: [[bench.base]] · [[bench.suites]] · [[bench.cqe]]
- `infer/`, `evalrun/`, `results/`: [[infer.backends]] · [[evalrun.run]] · [[results.store]] · [[results.report]]
- `train/`: [[train.common]] · [[train.sft]] · [[train.grpo]] · [[train.callbacks]]

**Data** — [[Data overview]] · [[DeepCAD and Text2CAD]] · [[Text-to-CadQuery dataset]] ·
[[CAD-Coder dataset]] · [[Synthetic parts]] · [[Dataset recipes]]

**Benchmarks** — [[Benchmarks overview]] · [[Metrics]] · [[Text2CAD-Bench]] ·
[[CadQueryEval]] · [[CADPrompt]] · [[DeepCAD held-out]] · [[Synth benchmark]]

**Tests** — [[Tests overview]] · [[test_geometry]] · [[test_sandbox]] · [[test_eval]] ·
[[test_build]] · [[test_synth]] · [[test_grpo_reward]] · [[test_callback]]

**Concepts** (short explanation + external resources)
- language models: [[LLMs and transformers]] · [[Tokenizers and chat templates]] · [[Decoding]] · [[Qwen3]]
- training: [[Supervised fine-tuning]] · [[Full fine-tuning vs LoRA]] · [[Mixed precision]] · [[Multi-GPU training]]
- reinforcement learning: [[Reinforcement learning for LLMs]] · [[GRPO]]
- inference: [[vLLM]]
- CAD and geometry: [[CadQuery and B-rep]] · [[Comparing 3D shapes]] · [[Rigid alignment]]
- evaluation: [[Evaluating code generation]] · [[Benchmark contamination]]
- engineering: [[Sandboxing untrusted code]] · [[Reproducible environments]] · [[Slurm]] · [[Experiment tracking]]
