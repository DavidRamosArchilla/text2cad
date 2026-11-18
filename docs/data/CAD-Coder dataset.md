---
tags: [data]
aliases: [cadcoder, CAD-Coder]
---
# CAD-Coder dataset (`cadcoder`)

Expert-level descriptions paired with CadQuery, from the closest prior work to this project.

- **Paper:** *CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric
  Reward*, NeurIPS 2025 — <https://arxiv.org/abs/2505.19713>. They did SFT + GRPO with a
  Chamfer-distance reward on Qwen2.5-7B: essentially the plan of this repository, one year
  earlier. Worth reading.
- **Download:** <https://huggingface.co/datasets/gudo7208/CAD-Coder>, pinned commit
  `cd716100` · code: <https://github.com/gudo7208/CAD-Coder>
- **Licence:** Apache-2.0.

## Files and what we use
| File | Samples | Used here |
|---|---|---|
| `cad_data_train_high.json` | 8 177 — verified, used for SFT in the paper | yes (`train_high`) |
| `cad_data_train_middle.json` | 66 534 — medium quality | prepared (`train_middle`), not in the default recipe |
| `cad_data_validation.json` | 8 817 | prepared (`val`) |
| `cad_data_train_cot.json`, `cad_data_test_cot.json` | chain-of-thought data | not used |
| `cad_data_train_all.json` | 156 954 — everything | not downloaded |

## What a row looks like
The user message is an *expert* Text2CAD description:
> Create a new coordinate system with the Euler angles set to [0.0, 0.0, -90.0] and the
> translation vector set to [-0.0174, 0.0174, 0.061]. On the first face, draw the first
> loop consisting of four lines: start at (0.0, 0.0) and draw a line to (0.7849, 0.0) …

The assistant answer is CadQuery whose final solid is called `r` (with `#show_object(r)`
comments). [[data.prepare]] strips the fixed request prefix ("Please based on the
following description, create a CAD-Query Code to generate a model:"), renames `r` to
`result`, and drops 8 117 exact duplicates between the high and middle files.

After verification: 8 116 high, 57 460 middle, 8 635 val. In `sft/v1`, 7 569 high rows
remain after removing objects that are held out ([[Dataset recipes]]).

## Why it is useful
Coordinate-level descriptions teach the model the mechanics of CadQuery sketches
(`moveTo`, `lineTo`, `close`, `extrude`, transforms) very explicitly. It is still
DeepCAD-only geometry in normalised units.
