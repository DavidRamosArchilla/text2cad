---
tags: [data]
aliases: [t2cq, Text-to-CadQuery, IEEE-T-ASE]
---
# Text-to-CadQuery dataset (`t2cq`)

The largest training corpus: Text2CAD descriptions paired with CadQuery scripts.

- **Paper:** *Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large
  Model Capabilities* (2025) — <https://arxiv.org/abs/2505.06507>
- **Download:** Hugging Face *model* repository `ricemonster/IEEE-T-ASE`
  (<https://huggingface.co/ricemonster/IEEE-T-ASE>), pinned commit `4091ce30`; we fetch
  `data/data_{train,val,test}.jsonl` (90 / 5 / 5 split by the authors).
- **How it was made:** the authors asked Gemini 2.0 Flash to write a CadQuery script for
  each [[DeepCAD and Text2CAD|Text2CAD]] description (≈ 170 000 scripts; 110 227 pairs
  in the JSONL files).
- **Licence:** not stated; derived from Text2CAD (CC BY-NC-SA 4.0).

## What a row looks like (raw)
```json
{"input": "Start by creating a new coordinate system with specific angles and translations. Then, draw a circular shape on the first face. … The final dimensions of the object will be 0.3 units in length and width, and 0.375 units in height.",
 "output": "import cadquery as cq\n…\nresult = part_1\ncq.exporters.export(result, './stlcq/0038/00380995.stl')"}
```
The descriptions are mid-level Text2CAD text (overall shape, part dimensions, sometimes
the modelling steps) in normalised units. The DeepCAD id is only visible in the export
path — [[data.prepare]] extracts it from there.

## After verification
| Split | Rows | Used as |
|---|---|---|
| train | 98 473 | SFT data (`sft/v1`, `sft/deepcad`) and part of the RL pool |
| val | 5 473 | held out; SFT validation rows |
| test | 5 472 | held out; the [[DeepCAD held-out]] benchmark samples 500 of them |

Dropped: 587 scripts that raise, 165 not watertight, 39 degenerate, a few timeouts and
unnormalisable scripts (`data/processed/t2cq/stats.json`).

## Things to keep in mind
- The scripts are Gemini's *interpretation* of the description; they were never checked
  against the original DeepCAD geometry. "Correct" here means "matches Gemini's script".
- The authors' own fine-tuned models score IoU ≈ 0.03–0.07 on Text2CAD-Bench
  ("Text2CADQuery" row in the paper) — a warning that this data alone does not transfer.
