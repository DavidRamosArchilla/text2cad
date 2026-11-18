---
tags: [module, verifier]
aliases: [codeutil.py, extract_code]
---
# cad.codeutil
**File:** `src/t2c/cad/codeutil.py` · **Libraries:** `ast`, `re` (standard library)

Text handling around scripts: get the code out of a model answer, refuse dangerous code
before running it, and normalise the reference scripts of the training corpora.

## `extract_code(text)` — model answer → script
1. Remove `<think>…</think>` reasoning (an unfinished `<think>` means no answer).
2. Prefer the **last** ```` ```python ```` block that mentions cadquery (models often show a
   draft and then a fixed version), then any fenced block, then an unterminated fence (the
   answer was cut off), then raw text starting at `import cadquery`.
3. Nothing found → `None` → the verifier reports `no_code`.

## `static_check(code)` — refuse before running
Parses the script with Python's `ast` and rejects (error type `unsafe_code`):
- imports of `subprocess`, `socket`, `shutil`, `urllib`, `requests`, `ctypes`,
  `multiprocessing`, `pickle`, `importlib`, … (about 35 modules);
- calls to `eval`, `exec`, `compile`, `__import__`, `input`, `exit`, …;
- `os.system`, `os.remove`, `os.kill`, `os.environ`, `sys.exit`, `sys.modules`, …

It also catches syntax errors (`syntax_error`). This is a guard against accidents; the
real protection is the process isolation in [[cad.sandbox]].

## `normalize_reference_script(code, final_hint=None)` — clean training targets
The downloaded corpora were written by other LLMs with different habits: they export STL
files to paths that do not exist, call `show_object`, or name the final solid `r`,
`assembly`, `part_1`… This function makes every script follow one contract — *start with
`import cadquery as cq`, final solid in `result`, nothing else*:

1. drop export / visualisation lines (`cq.exporters.export(...)`, `show(...)`, …);
2. find the final solid: the variable that was exported, else the dataset's convention
   (`final_hint`, e.g. `r` for CAD-Coder), else an existing `result`, else the last
   top-level assignment;
3. rename it to `result` **by syntax-tree position**, so `part_1_width` or a comment that
   contains `part_1` are left untouched;
4. return `None` if this cannot be done safely (the sample is dropped).

**Used by:** [[evalrun.run]], [[train.grpo]] (extract), [[cad.sandbox]] (static check),
[[data.prepare]], [[bench.suites]] (normalise).
**Tested in:** [[test_sandbox]] (`test_extract_code`, `test_normalize_reference_script`, `test_error_taxonomy`).
