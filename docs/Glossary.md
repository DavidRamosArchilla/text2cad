---
tags: [index]
aliases: [Terms, Vocabulary]
---
# Glossary

Short definitions of words used across the code and these notes. Follow the link for more.

| Term | Meaning |
|---|---|
| **advantage** | In [[GRPO]]: how much better one sample's reward is than the average of its group, divided by the group's standard deviation. Positive → make that answer more likely. |
| **API model** | A model reached over HTTP (OpenAI, OpenRouter…) instead of loaded locally. Stage `api` in results. See [[infer.backends]]. |
| **B-rep** | Boundary representation: a solid stored as its exact faces, edges and vertices. What CadQuery builds. See [[CadQuery and B-rep]]. |
| **backend** | The object that turns chat conversations into model answers: `vllm`, `hf` or `openai`. See [[infer.backends]]. |
| **base model** | A model as downloaded, before our fine-tuning (stage `base`). |
| **bf16 / fp32** | 16-bit "brain float" and 32-bit floating point. Training computes in bf16 but keeps weights in fp32. See [[Mixed precision]]. |
| **canary string** | A unique text placed in benchmark files so that anyone can detect (and filter) them in training data. See [[Benchmark contamination]]. |
| **chat template** | The tokenizer's rule for turning a list of `{role, content}` messages into the exact text the model reads. See [[Tokenizers and chat templates]]. |
| **Chamfer distance (CD)** | Average distance between two point clouds sampled on two surfaces. Lower is better. See [[Metrics]]. |
| **checkpoint** | Saved model weights (plus optimizer state) at some training step: `outputs/train/<run>/checkpoint-N/`. |
| **colocate** | vLLM running on the same GPU as the training model during GRPO, sharing memory. See [[vLLM]]. |
| **completion** | The part of a training example the model must learn to produce (here: the fenced CadQuery script). |
| **DeepCAD uid** | The 8-digit id of a DeepCAD model (e.g. `00357061`). Used to stop benchmark objects leaking into training. See [[Dataset recipes]]. |
| **epoch** | One pass over the training set. |
| **executor** | `CadExecutor`, the pool that runs scripts in sandboxed processes. See [[cad.sandbox]]. |
| **geo / seq prompt** | Two description styles: *geo* describes the finished shape, *seq* gives the modelling steps. Used in [[Text2CAD-Bench]] and [[Synthetic parts]]. |
| **ground truth (GT)** | The correct solid for a task, stored as a triangle mesh `gt/*.npz`. |
| **group** | In GRPO: the N answers sampled for the same prompt (here N = 8). In benchmarks: a subset of samples (e.g. `L2/geo`) reported separately. |
| **harness error** | A failure of *our* scoring code (not the model's fault). Excluded from metrics and printed as a warning. |
| **headline metric** | The one number per benchmark shown on the scoreboard. See [[Reading the report]]. |
| **holdout** | Objects deliberately kept out of training (benchmark objects, validation/test splits). See [[Dataset recipes]]. |
| **IoU** | Intersection over union of two solids, computed on voxels. 1 = identical. See [[Metrics]]. |
| **IoU_all** | IoU averaged over *all* samples, invalid answers counting as 0. The fairest single number. |
| **IR / IR@1** | Invalid rate: share of answers that do not produce a valid solid, after retries / on the first attempt. |
| **LoRA** | Low-rank adapters, a cheaper alternative to full fine-tuning (not used here). See [[Full fine-tuning vs LoRA]]. |
| **mesh** | Triangles approximating a surface. Solids are converted to meshes to be compared. See [[Comparing 3D shapes]]. |
| **model spec** | `ModelSpec`: name, path, backend, prompt style, sampling settings and lineage of a model to evaluate. See [[infer.backends]]. |
| **native / protocol prompt** | *native*: the short system prompt our fine-tuned models are trained with. *protocol*: the benchmark's own published prompt (for base and API models). See [[prompts]]. |
| **policy** | In RL, the model being trained (it "decides" which tokens to produce). |
| **recipe** | A YAML file in `configs/data/` that says which data goes into a training set. See [[Dataset recipes]]. |
| **retries** | Extra attempts where the model sees its own failed script and the error message. Text2CAD-Bench allows 3. See [[evalrun.run]]. |
| **reward** | The number GRPO maximises; here from the verifier. See [[cad.reward]]. |
| **rollout** | One sampled answer generated during RL training. |
| **run** | One training run (`outputs/train/<run_name>/`) or one evaluation run (`results/runs/<run_id>/`). |
| **SFT** | Supervised fine-tuning: learn to reproduce reference answers. See [[Supervised fine-tuning]]. |
| **stage (verifier)** | How far a script got: 0 nothing runnable, 1 parsed but crashed, 2 ran but no solid, 3 valid solid. See [[cad.sandbox]]. |
| **stage (model)** | Lineage tag in results: `base`, `sft`, `grpo`, `api`. |
| **step** | One optimiser update. |
| **tessellation** | Converting an exact B-rep solid into a triangle mesh. |
| **thinking mode** | Qwen3 can write hidden reasoning in `<think>…</think>` before answering. Switched off here by default. See [[Qwen3]]. |
| **token** | The unit an LLM reads and writes (a word piece). See [[LLMs and transformers]]. |
| **verifier** | The code that runs a script and compares the result with ground truth (`src/t2c/cad/`). |
| **voxel** | A small cube in a 3D grid; a solid becomes a grid of filled / empty voxels for IoU. |
| **watertight** | A mesh with no holes: every edge is shared by exactly two triangles. Needed for volume and IoU to make sense. |
| **zygote** | A pre-loaded process that forks a fresh child for every script, so each one starts in ~10 ms. See [[cad.sandbox]]. |
