---
tags: [data]
aliases: [DeepCAD, Text2CAD]
---
# DeepCAD and Text2CAD

The origin of most of the data. Worth knowing because their properties explain the
strengths and weaknesses of the training corpora.

## DeepCAD (2021)
- About 178 000 CAD models taken from Onshape (via the ABC dataset), each stored as a
  **sequence of CAD commands**: 2D sketches (lines, arcs, circles) and **extrusions**. No
  fillets, chamfers, revolves, sweeps or lofts.
- Coordinates are **normalised**: parts fit roughly in a unit cube, so dimensions look
  like `0.75` or `0.0204` "units", not millimetres.
- Every model has an 8-digit id, e.g. `00357061`. This **uid** follows the object through
  every derived dataset, and is how [[Dataset recipes]] keeps benchmark objects out of training.
- Paper: Wu, Xiao, Zheng, *DeepCAD: A Deep Generative Network for Computer-Aided Design
  Models*, ICCV 2021 — <https://arxiv.org/abs/2105.09492> · code:
  <https://github.com/ChrisWu1997/DeepCAD>

## Text2CAD (2024)
- Adds **text descriptions** to ~170 000 DeepCAD models at four levels, from *abstract*
  ("a cylindrical object") to *expert* (coordinates of every sketch line, Euler angles,
  extrusion depths). The descriptions were generated automatically by vision-language and
  language models from renders and command sequences.
- Licence CC BY-NC-SA 4.0 (non-commercial).
- Paper: Khan et al., *Text2CAD: Generating Sequential CAD Models from Beginner-to-Expert
  Level Text Prompts*, NeurIPS 2024 — <https://arxiv.org/abs/2409.17106> · data:
  <https://huggingface.co/datasets/SadilKhan/Text2CAD> · code:
  <https://github.com/SadilKhan/Text2CAD>

Text2CAD's own model generates DeepCAD command sequences, not code. Others then asked
LLMs to translate the same descriptions into CadQuery.

## What descends from them here
| Dataset / benchmark | How |
|---|---|
| [[Text-to-CadQuery dataset]] | Text2CAD descriptions → CadQuery written by Gemini 2.0 Flash |
| [[CAD-Coder dataset]] | expert-level Text2CAD descriptions → CadQuery |
| [[CADPrompt]] | 200 DeepCAD objects with new, human-written prompts |
| [[DeepCAD held-out]] | 500 objects of the Text-to-CadQuery test split |

## Consequences
- **Everything is sketch-and-extrude in normalised units** → [[Synthetic parts]] fill the
  millimetre / richer-operation gap.
- **The same object appears in several datasets** (95 of CADPrompt's 200 objects are in
  the Text-to-CadQuery training split) → leakage control by uid in [[data.build]].
- **The descriptions are machine-written**, long and regular; real prompts (Text2CAD-Bench,
  CadQueryEval, CADPrompt) read differently.
