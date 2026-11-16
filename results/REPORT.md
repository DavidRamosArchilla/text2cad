# Text → CadQuery benchmark report

_Generated 2026-11-15 22:43 from 40 recorded runs. Rebuild with `t2c report`._

## Scoreboard

_Headline metric per benchmark, each at its protocol's error-feedback retry budget (t2cb: 3, cqe: 0, cadprompt: 0, deepcad: 0, synth: 0). Parentheses: change versus the same model before fine-tuning. IR = invalid rate; IoU_all counts invalid outputs as 0._

| Model | Stage | Prompt | t2cb IR ↓ | cqe pass ↑ | cqe IoU_all ↑ | cadprompt IoU_all ↑ | deepcad IoU_all ↑ | synth IoU_all ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | grpo | native | 23.5% (−67.5pp) | 24.0% (+24.0pp) | 0.425 (+0.361) | 0.229 (+0.136) | 0.620 (+0.595) | 1.000 (+0.942) |
| **pilot-grpo-qwen3-0.6b** | grpo | native | 34.8% (−65.2pp) | 12.0% (+12.0pp) | 0.268 | 0.182 (+0.181) | 0.579 (+0.578) | 0.993 (+0.991) |
| **pilot-sft-qwen3-0.6b** | sft | native | 36.4% (−63.6pp) | 20.0% (+20.0pp) | 0.322 | 0.172 (+0.171) | 0.565 (+0.564) | 0.990 (+0.989) |
| **sft-qwen3-1.7b-v1** | sft | native | 24.2% (−66.9pp) | 20.0% (+20.0pp) | 0.366 (+0.301) | 0.206 (+0.112) | 0.610 (+0.585) | 1.000 (+0.942) |
| **qwen3-0.6b** | base | protocol | 100.0% | 0.0% | – | 0.000 | 0.001 | 0.001 |
| **qwen3-1.7b** | base | protocol | 91.1% | 0.0% | 0.064 | 0.093 | 0.025 | 0.058 |
| **qwen3-4b** | base | protocol | 74.2% | 4.0% | 0.040 | 0.125 | 0.111 | 0.076 |
| **qwen3-8b** | base | protocol | 61.9% | 12.0% | 0.283 | 0.129 | 0.106 | 0.207 |

## t2cb

Text2CAD-Bench public sample: 151 parts x {geo, seq} prompts, L1-L3 (IR only unless GT is supplied)

### Leaderboard

_Rows marked 'published' — Text2CAD-Bench, arXiv 2605.18430, Table 1 (and the dataset card's leaderboard). Published on the full 600-part benchmark with the authors' prompt (5 in-context examples) and up to 3 error-feedback retries. We can only run the public 151-part prompt sample, and only IR until ground truth is obtained from the authors. Scores date from early 2026._

| Model | Stage | Protocol | IR ↓ | IR@1 ↓ | n | Date |
|---|---:|---:|---:|---:|---:|---:|
| Text2CAD (specialist) | published | see note | 7.0% | – |  |  |
| **grpo-qwen3-1.7b-v1** | grpo | native, 3 retr. | 23.5% | 26.5% | 302 | 2026-11-15 |
| **sft-qwen3-1.7b-v1** | sft | native, 3 retr. | 24.2% | 26.5% | 302 | 2026-11-15 |
| DeepSeek-V3.2 | published | see note | 29.7% | – |  |  |
| GPT-5.2 | published | see note | 30.6% | – |  |  |
| **pilot-grpo-qwen3-0.6b** | grpo | native, 3 retr. | 34.8% | 34.8% | 302 | 2026-11-15 |
| GLM-4.7 | published | see note | 35.0% | – |  |  |
| **pilot-sft-qwen3-0.6b** | sft | native, 3 retr. | 36.4% | 36.8% | 302 | 2026-11-15 |
| Claude-4.5-Sonnet | published | see note | 41.3% | – |  |  |
| Qwen3-max | published | see note | 43.2% | – |  |  |
| Text2CADQuery (specialist) | published | see note | 51.0% | – |  |  |
| CADFusion (specialist) | published | see note | 60.5% | – |  |  |
| **qwen3-8b** | base | protocol, 3 retr. | 61.9% | 73.5% | 302 | 2026-11-15 |
| **qwen3-4b** | base | protocol, 3 retr. | 74.2% | 84.4% | 302 | 2026-11-15 |
| **qwen3-1.7b** | base | protocol, 3 retr. | 91.1% | 94.0% | 302 | 2026-11-15 |
| **qwen3-0.6b** | base | protocol, 3 retr. | 100.0% | 100.0% | 302 | 2026-11-15 |

### By group — IR

| Model | Protocol | L1/geo | L1/seq | L2/geo | L2/seq | L3/geo | L3/seq |
|---|---:|---:|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 3 retr. | 11.7% | 18.3% | 21.3% | 26.2% | 33.3% | 46.7% |
| **pilot-grpo-qwen3-0.6b** | native, 3 retr. | 16.7% | 38.3% | 31.1% | 32.8% | 46.7% | 63.3% |
| **pilot-sft-qwen3-0.6b** | native, 3 retr. | 28.3% | 25.0% | 26.2% | 41.0% | 56.7% | 66.7% |
| **sft-qwen3-1.7b-v1** | native, 3 retr. | 11.7% | 18.3% | 21.3% | 23.0% | 43.3% | 50.0% |
| **qwen3-0.6b** | protocol, 3 retr. | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **qwen3-1.7b** | protocol, 3 retr. | 85.0% | 85.0% | 86.9% | 98.4% | 100.0% | 100.0% |
| **qwen3-4b** | protocol, 3 retr. | 60.0% | 56.7% | 80.3% | 83.6% | 86.7% | 93.3% |
| **qwen3-8b** | protocol, 3 retr. | 40.0% | 43.3% | 68.9% | 68.9% | 76.7% | 100.0% |
| GPT-5.2 | published | 11.1% | 19.5% | 20.0% | 31.6% | 68.0% | 74.0% |
| Claude-4.5-Sonnet | published | 20.3% | 25.3% | 43.7% | 53.5% | 70.0% | 75.0% |
| Qwen3-max | published | 21.4% | 30.1% | 33.4% | 49.1% | 92.0% | 93.0% |
| Text2CADQuery (specialist) | published | 44.0% | – | 67.0% | – | 80.0% | – |

### Why outputs were invalid (after retries)

| Model | Protocol | degenerate | no_result | no_solid | runtime_error | syntax_error |
|---|---:|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 3 retr. | 0.0% | 0.0% | 2.6% | 20.5% | 0.3% |
| **pilot-grpo-qwen3-0.6b** | native, 3 retr. | 0.3% | 1.0% | 2.3% | 28.5% | 2.6% |
| **pilot-sft-qwen3-0.6b** | native, 3 retr. | 0.7% | 0.7% | 2.0% | 26.2% | 7.0% |
| **sft-qwen3-1.7b-v1** | native, 3 retr. | 0.0% | 0.3% | 2.0% | 19.2% | 2.6% |
| **qwen3-0.6b** | protocol, 3 retr. | 0.0% | 0.0% | 0.0% | 77.5% | 22.5% |
| **qwen3-1.7b** | protocol, 3 retr. | 0.0% | 0.0% | 0.0% | 72.2% | 18.9% |
| **qwen3-4b** | protocol, 3 retr. | 0.0% | 0.0% | 0.3% | 64.6% | 9.3% |
| **qwen3-8b** | protocol, 3 retr. | 0.0% | 0.0% | 0.0% | 51.3% | 10.6% |

## cqe

CadQueryEval: 25 mm-scale parts, pass/fail on six geometric checks; frontier leaderboard available

### Leaderboard

_Rows marked 'published' — CadQueryEval leaderboard (drwahl/cadqueryeval), recomputed from the per-attempt table shipped with the dataset. Same tasks and prompt as our 'protocol' runs; our scorer re-implements the official checks without open3d and agrees with the published verdicts on a validation sample (`t2c bench validate-cqe`). Rows are loaded automatically from the dataset. Showing the top 10 and the Qwen-family entries of 91 published models._

| Model | Stage | Protocol | Pass ↑ | IR ↓ | IR@1 ↓ | IoU_all ↑ | IoU (valid) ↑ | CD×10³ ↓ | dim err ↓ | n | Date |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-opus-5.5 | published | see note | 100.0% | 0.0% | – | – | – | – | – |  | 2026-09-22 |
| gpt-5.6-sol-pro | published | see note | 100.0% | 0.0% | – | – | – | – | – |  | 2026-07-09 |
| gpt-6-astra | published | see note | 100.0% | 0.0% | – | – | – | – | – |  | 2026-09-04 |
| gpt-6-sol | published | see note | 100.0% | 0.0% | – | – | – | – | – |  | 2026-09-22 |
| gemini-3.8-flash | published | see note | 96.0% | 0.0% | – | – | – | – | – |  | 2026-09-02 |
| gpt-6-luna | published | see note | 96.0% | 0.0% | – | – | – | – | – |  | 2026-09-22 |
| glm-5.3-flash | published | see note | 92.0% | 8.0% | – | – | – | – | – |  | 2026-08-28 |
| gemini-3.1-pro-preview | published | see note | 88.0% | 0.0% | – | – | – | – | – |  | 2026-02-20 |
| gpt-5.6-luna | published | see note | 88.0% | 8.0% | – | – | – | – | – |  | 2026-07-09 |
| gpt-5.6-luna-pro | published | see note | 88.0% | 8.0% | – | – | – | – | – |  | 2026-07-09 |
| qwen3.8-max | published | see note | 88.0% | 4.0% | – | – | – | – | – |  | 2026-08-03 |
| qwen3.7-max | published | see note | 80.0% | 12.0% | – | – | – | – | – |  | 2026-05-22 |
| qwen3.7-plus | published | see note | 56.0% | 24.0% | – | – | – | – | – |  | 2026-06-03 |
| qwen3.8-flash | published | see note | 48.0% | 24.0% | – | – | – | – | – |  | 2026-08-28 |
| qwen3.6-plus | published | see note | 44.0% | 36.0% | – | – | – | – | – |  | 2026-04-23 |
| **grpo-qwen3-1.7b-v1** | grpo | native, 0 retr. | 24.0% | 36.0% | 36.0% | 0.425 | 0.664 | 84.472 | 0.000 | 25 | 2026-11-15 |
| **pilot-sft-qwen3-0.6b** | sft | native, 0 retr. | 20.0% | 24.0% | 24.0% | 0.322 | 0.423 | 184.112 | 0.098 | 25 | 2026-11-15 |
| **sft-qwen3-1.7b-v1** | sft | native, 0 retr. | 20.0% | 36.0% | 36.0% | 0.366 | 0.571 | 116.348 | 0.048 | 25 | 2026-11-15 |
| qwen3.7-flash | published | see note | 20.0% | 76.0% | – | – | – | – | – |  | 2026-07-29 |
| **qwen3-8b** | base | protocol, 0 retr. | 12.0% | 52.0% | 52.0% | 0.283 | 0.590 | 132.339 | 0.000 | 25 | 2026-11-15 |
| **pilot-grpo-qwen3-0.6b** | grpo | native, 0 retr. | 12.0% | 36.0% | 36.0% | 0.268 | 0.419 | 180.863 | 0.377 | 25 | 2026-11-15 |
| **qwen3-4b** | base | protocol, 0 retr. | 4.0% | 96.0% | 96.0% | 0.040 | 0.999 | 7.459 | 0.000 | 25 | 2026-11-15 |
| **qwen3-1.7b** | base | protocol, 0 retr. | 0.0% | 92.0% | 92.0% | 0.064 | 0.804 | 30.875 | 0.058 | 25 | 2026-11-15 |
| **qwen3-0.6b** | base | protocol, 0 retr. | 0.0% | 100.0% | 100.0% | – | – | – | – | 25 | 2026-11-15 |

### Why outputs were invalid (after retries)

| Model | Protocol | no_output_file | runtime_error | syntax_error |
|---|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 0 retr. | 0.0% | 36.0% | 0.0% |
| **pilot-grpo-qwen3-0.6b** | native, 0 retr. | 0.0% | 36.0% | 0.0% |
| **pilot-sft-qwen3-0.6b** | native, 0 retr. | 0.0% | 24.0% | 0.0% |
| **sft-qwen3-1.7b-v1** | native, 0 retr. | 0.0% | 36.0% | 0.0% |
| **qwen3-0.6b** | protocol, 0 retr. | 0.0% | 60.0% | 40.0% |
| **qwen3-1.7b** | protocol, 0 retr. | 0.0% | 80.0% | 12.0% |
| **qwen3-4b** | protocol, 0 retr. | 4.0% | 92.0% | 0.0% |
| **qwen3-8b** | protocol, 0 retr. | 0.0% | 48.0% | 4.0% |

## cadprompt

CADPrompt: 200 DeepCAD objects, human-written prompts ('measured' = with dimensions, 'abstract' = without)

### Leaderboard

_Rows marked 'published' — CADPrompt / CADCodeVerify, arXiv 2410.05340 (ICLR 2025), Table 2, single-shot 'Generated' rows. Only the compile rate is comparable (reported here as IR = 1 - compile rate). The paper's geometric metrics use ICP alignment, 1000 points and medians, so they are not listed. 2024-era models._

| Model | Stage | Protocol | IR ↓ | IR@1 ↓ | IoU_all ↑ | IoU (valid) ↑ | CD×10³ ↓ | dim err ↓ | n | Date |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | grpo | native, 0 retr. | 1.7% | 1.7% | 0.229 | 0.233 | 228.646 | 0.672 | 400 | 2026-11-15 |
| **sft-qwen3-1.7b-v1** | sft | native, 0 retr. | 2.2% | 2.2% | 0.206 | 0.210 | 240.966 | 0.771 | 400 | 2026-11-15 |
| **pilot-grpo-qwen3-0.6b** | grpo | native, 0 retr. | 3.7% | 3.7% | 0.182 | 0.189 | 259.949 | 0.880 | 400 | 2026-11-15 |
| **pilot-sft-qwen3-0.6b** | sft | native, 0 retr. | 7.0% | 7.0% | 0.172 | 0.185 | 264.871 | 0.971 | 400 | 2026-11-15 |
| **qwen3-8b** | base | protocol, 0 retr. | 55.5% | 55.5% | 0.129 | 0.291 | 227.994 | 3.364 | 400 | 2026-11-15 |
| **qwen3-4b** | base | protocol, 0 retr. | 61.3% | 61.3% | 0.125 | 0.323 | 203.437 | 1.746 | 400 | 2026-11-15 |
| **qwen3-1.7b** | base | protocol, 0 retr. | 75.8% | 75.8% | 0.093 | 0.384 | 163.399 | 0.611 | 400 | 2026-11-15 |
| **qwen3-0.6b** | base | protocol, 0 retr. | 99.8% | 99.8% | 0.000 | 0.122 | 124.297 | 1.453 | 400 | 2026-11-15 |
| GPT-4 (zero-shot) | published | see note | 8.0% | – | – | – | – | – |  |  |
| GPT-4 (few-shot) | published | see note | 4.0% | – | – | – | – | – |  |  |
| Gemini (zero-shot) | published | see note | 15.0% | – | – | – | – | – |  |  |
| CodeLlama (zero-shot) | published | see note | 35.5% | – | – | – | – | – |  |  |

### By group — IoU_all

| Model | Protocol | abstract | measured |
|---|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 0 retr. | 0.157 | 0.301 |
| **pilot-grpo-qwen3-0.6b** | native, 0 retr. | 0.082 | 0.281 |
| **pilot-sft-qwen3-0.6b** | native, 0 retr. | 0.078 | 0.265 |
| **sft-qwen3-1.7b-v1** | native, 0 retr. | 0.124 | 0.287 |
| **qwen3-0.6b** | protocol, 0 retr. | – | 0.001 |
| **qwen3-1.7b** | protocol, 0 retr. | 0.032 | 0.154 |
| **qwen3-4b** | protocol, 0 retr. | 0.058 | 0.192 |
| **qwen3-8b** | protocol, 0 retr. | 0.072 | 0.186 |

### Why outputs were invalid (after retries)

| Model | Protocol | degenerate | no_solid | runtime_error | syntax_error |
|---|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 0 retr. | 0.2% | 0.0% | 1.5% | 0.0% |
| **pilot-grpo-qwen3-0.6b** | native, 0 retr. | 0.0% | 0.5% | 2.0% | 1.2% |
| **pilot-sft-qwen3-0.6b** | native, 0 retr. | 0.0% | 1.5% | 4.0% | 1.5% |
| **sft-qwen3-1.7b-v1** | native, 0 retr. | 0.2% | 0.5% | 1.2% | 0.2% |
| **qwen3-0.6b** | protocol, 0 retr. | 0.0% | 0.0% | 88.8% | 11.0% |
| **qwen3-1.7b** | protocol, 0 retr. | 0.0% | 0.5% | 69.5% | 5.8% |
| **qwen3-4b** | protocol, 0 retr. | 0.0% | 0.2% | 59.8% | 1.2% |
| **qwen3-8b** | protocol, 0 retr. | 0.2% | 0.2% | 53.5% | 1.5% |

## deepcad

500 held-out Text2CAD/DeepCAD descriptions (test split of Text-to-CadQuery); GT = executed reference script

### Leaderboard

| Model | Stage | Protocol | IR ↓ | IR@1 ↓ | IoU_all ↑ | IoU (valid) ↑ | CD×10³ ↓ | dim err ↓ | n | Date |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | grpo | native, 0 retr. | 0.4% | 0.4% | 0.620 | 0.622 | 93.102 | 0.000 | 500 | 2026-11-15 |
| **sft-qwen3-1.7b-v1** | sft | native, 0 retr. | 1.6% | 1.6% | 0.610 | 0.619 | 92.484 | 0.000 | 500 | 2026-11-15 |
| **pilot-grpo-qwen3-0.6b** | grpo | native, 0 retr. | 0.4% | 0.4% | 0.579 | 0.581 | 99.373 | 0.002 | 500 | 2026-11-15 |
| **pilot-sft-qwen3-0.6b** | sft | native, 0 retr. | 4.4% | 4.4% | 0.565 | 0.591 | 101.404 | 0.001 | 500 | 2026-11-15 |
| **qwen3-4b** | base | protocol, 0 retr. | 66.0% | 66.0% | 0.111 | 0.328 | 171.786 | 0.408 | 500 | 2026-11-15 |
| **qwen3-8b** | base | protocol, 0 retr. | 67.0% | 67.0% | 0.106 | 0.320 | 178.325 | 0.357 | 500 | 2026-11-15 |
| **qwen3-1.7b** | base | protocol, 0 retr. | 91.0% | 91.0% | 0.025 | 0.274 | 169.462 | 0.288 | 500 | 2026-11-15 |
| **qwen3-0.6b** | base | protocol, 0 retr. | 99.6% | 99.6% | 0.001 | 0.222 | 216.486 | 0.467 | 500 | 2026-11-15 |

### Why outputs were invalid (after retries)

| Model | Protocol | degenerate | no_result | runtime_error | syntax_error |
|---|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 0 retr. | 0.0% | 0.0% | 0.2% | 0.2% |
| **pilot-grpo-qwen3-0.6b** | native, 0 retr. | 0.0% | 0.0% | 0.4% | 0.0% |
| **pilot-sft-qwen3-0.6b** | native, 0 retr. | 0.0% | 0.4% | 2.8% | 1.2% |
| **sft-qwen3-1.7b-v1** | native, 0 retr. | 0.2% | 0.0% | 1.0% | 0.4% |
| **qwen3-0.6b** | protocol, 0 retr. | 0.0% | 0.2% | 84.2% | 15.2% |
| **qwen3-1.7b** | protocol, 0 retr. | 0.0% | 0.0% | 89.2% | 1.8% |
| **qwen3-4b** | protocol, 0 retr. | 0.0% | 0.0% | 65.4% | 0.6% |
| **qwen3-8b** | protocol, 0 retr. | 0.0% | 0.0% | 65.0% | 2.0% |

## synth

Held-out procedural parts in millimetres (plates, flanges, shafts, brackets, ...): exact GT, geo + seq prompts

### Leaderboard

| Model | Stage | Protocol | IR ↓ | IR@1 ↓ | IoU_all ↑ | IoU (valid) ↑ | CD×10³ ↓ | dim err ↓ | n | Date |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sft-qwen3-1.7b-v1** | sft | native, 0 retr. | 0.0% | 0.0% | 1.000 | 1.000 | 9.008 | 0.000 | 800 | 2026-11-15 |
| **grpo-qwen3-1.7b-v1** | grpo | native, 0 retr. | 0.0% | 0.0% | 1.000 | 1.000 | 9.015 | 0.000 | 800 | 2026-11-15 |
| **pilot-grpo-qwen3-0.6b** | grpo | native, 0 retr. | 0.1% | 0.1% | 0.993 | 0.994 | 9.477 | 0.000 | 800 | 2026-11-15 |
| **pilot-sft-qwen3-0.6b** | sft | native, 0 retr. | 0.2% | 0.2% | 0.990 | 0.993 | 9.673 | 0.000 | 800 | 2026-11-15 |
| **qwen3-8b** | base | protocol, 0 retr. | 58.5% | 58.5% | 0.207 | 0.500 | 142.053 | 0.108 | 800 | 2026-11-15 |
| **qwen3-4b** | base | protocol, 0 retr. | 84.4% | 84.4% | 0.076 | 0.484 | 137.170 | 0.055 | 800 | 2026-11-15 |
| **qwen3-1.7b** | base | protocol, 0 retr. | 89.9% | 89.9% | 0.058 | 0.570 | 126.384 | 0.070 | 800 | 2026-11-15 |
| **qwen3-0.6b** | base | protocol, 0 retr. | 99.9% | 99.9% | 0.001 | 0.989 | 7.324 | 0.000 | 800 | 2026-11-15 |

### By group — IoU_all

| Model | Protocol | beam_profile | boss_plate | cbore_plate | flange | gusset_bracket | l_bracket | loft | open_box | plate_holes | polar_pattern | polygon_prism | revolved | slotted_block | stepped_shaft | swept_elbow | tube_flange | wedge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 0 retr. | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| **pilot-grpo-qwen3-0.6b** | native, 0 retr. | 0.971 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 | 1.000 | 1.000 | 0.977 | 0.963 | 1.000 | 1.000 | 1.000 | 1.000 |
| **pilot-sft-qwen3-0.6b** | native, 0 retr. | 0.954 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.986 | 1.000 | 1.000 | 0.980 | 0.935 | 1.000 | 1.000 | 0.987 | 1.000 |
| **sft-qwen3-1.7b-v1** | native, 0 retr. | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| **qwen3-0.6b** | protocol, 0 retr. | – | – | – | – | – | – | – | – | 0.021 | – | – | – | – | – | – | – | – |
| **qwen3-1.7b** | protocol, 0 retr. | – | 0.109 | 0.190 | – | – | 0.000 | – | 0.061 | 0.136 | – | 0.100 | 0.020 | 0.100 | 0.109 | – | 0.150 | – |
| **qwen3-4b** | protocol, 0 retr. | 0.021 | 0.171 | 0.064 | 0.057 | 0.020 | 0.004 | – | 0.071 | 0.229 | 0.028 | 0.056 | 0.110 | 0.245 | 0.139 | – | 0.045 | 0.018 |
| **qwen3-8b** | protocol, 0 retr. | 0.034 | 0.343 | 0.521 | 0.021 | 0.018 | 0.035 | – | 0.093 | 0.514 | 0.034 | 0.142 | 0.118 | 0.466 | 0.443 | 0.001 | 0.584 | 0.137 |

### Why outputs were invalid (after retries)

| Model | Protocol | degenerate | no_solid | runtime_error | syntax_error |
|---|---:|---:|---:|---:|---:|
| **grpo-qwen3-1.7b-v1** | native, 0 retr. | 0.0% | 0.0% | 0.0% | 0.0% |
| **pilot-grpo-qwen3-0.6b** | native, 0 retr. | 0.0% | 0.0% | 0.1% | 0.0% |
| **pilot-sft-qwen3-0.6b** | native, 0 retr. | 0.0% | 0.0% | 0.2% | 0.0% |
| **sft-qwen3-1.7b-v1** | native, 0 retr. | 0.0% | 0.0% | 0.0% | 0.0% |
| **qwen3-0.6b** | protocol, 0 retr. | 0.0% | 0.0% | 58.5% | 41.4% |
| **qwen3-1.7b** | protocol, 0 retr. | 0.0% | 0.1% | 86.8% | 3.0% |
| **qwen3-4b** | protocol, 0 retr. | 0.0% | 0.1% | 81.6% | 2.6% |
| **qwen3-8b** | protocol, 0 retr. | 0.2% | 1.2% | 48.1% | 8.9% |
