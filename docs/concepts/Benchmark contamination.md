---
tags: [concept]
aliases: [contamination, leakage, canary]
---
# Benchmark contamination

If benchmark items — or the same objects described differently — end up in training
data, the benchmark measures **memory** instead of skill, and nothing in the numbers tells
you. Three ways it can happen in this project, and what is done about each:

| Risk | Defence |
|---|---|
| The same DeepCAD object in a training corpus and a benchmark (95 of CADPrompt's 200 objects are in the Text-to-CadQuery training split) | recipes hold out objects **by DeepCAD id**, across all corpora ([[Dataset recipes]], tested in [[test_build]]) |
| Training on a benchmark's own files | no recipe reads `t2cb`, `cqe` or `cadprompt` data; CadQueryEval files carry a **canary string** so they can be detected and filtered anywhere |
| The benchmark was in the base model's pre-training data | cannot be controlled; Text2CAD-Bench keeps its ground truth private for this reason, and CadQueryEval (Jan 2026) is newer than Qwen3 (2025) |

A **canary string** is a unique GUID placed in benchmark files with the request "never
include in training corpora"; anyone building a training set can search for it, and a
model that can reproduce it has seen the data.

## Learn more
- BIG-bench's canary convention: <https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/training_on_test_set/README.md>
- Sainz et al., *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination
  for each Benchmark* (2023): <https://arxiv.org/abs/2310.18018>
