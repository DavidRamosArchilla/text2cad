---
tags: [module, data]
aliases: [synth.py, generator, t2c data synth]
---
# data.synth
**File:** `src/t2c/data/synth.py` (≈ 1 200 lines, most of it prompt text) · **Libraries:** `random`, `math`, `hashlib` + [[cad.sandbox]]
**Data:** [[Synthetic parts]] · **Benchmark:** [[Synth benchmark]]

Procedural generator of millimetre-scale mechanical parts. Each part comes with a
CadQuery script and two descriptions, all consistent with each other and with the
geometry the script really builds. Why it exists: the public corpora are DeepCAD parts in
normalised units, while the target benchmarks describe real parts in millimetres — see
[[Synthetic parts]].

## Structure
- **`Part`** (dataclass): `family`, `code`, `geo` (describes the finished shape), `seq`
  (modelling steps), `bbox` (intended extents), `volume` (closed-form, when known), `dims`
  (key dimensions both texts must state).
- **17 family functions** `(rng) -> Part`, e.g. `plate_holes`, `flange`, `l_bracket`,
  `swept_elbow` — the full list is in [[Synthetic parts]]. They draw "nice" engineering
  values (`nice(rng, lo, hi, step)`), raise `Reject` when the numbers do not fit together
  (a hole larger than the plate…), and assemble code and text.
- **`V`** (code style): each script randomly uses named variables (`hole_d = 6`) or inline
  numbers, and different idioms (`box` vs `rect().extrude()`, `hole()` vs `cutThruAll()`),
  so the model does not learn one fixed template.
- **`Txt`** (text style): units written as "mm"/"millimeters", decimals or not, radius or
  diameter, several phrasings per sentence.
- **`make_part(family, rng)`** retries until a family does not reject.
- **`check_part(part, result)`**: after execution, keep the part only if it is one
  watertight solid whose bounding box matches the intended one (within 0.4 % of the part's
  largest dimension) and whose volume matches the closed-form value (±0.3 %) when there is one.

## `generate_dataset(n_train=20000, n_val=500, n_test=400, seed=0)`
- Candidate *i* of a split is generated from the random seed `"{seed}/{split}/{i}"` —
  fully reproducible and independent of the other splits.
- Test is generated first, then val, then train, so test/val never change when you ask for
  more training data.
- Any script already used (same hash) is skipped → the three splits are disjoint.
- Every candidate is executed and checked; ≈ 100 % pass for every family.
- Writes `data/processed/synth/{train,val,test}.jsonl`, **two rows per part** (`geo` and
  `seq`, same code and `uid`), and `stats.json` with per-family yield. uids start with
  `t`/`v`/`x` for train/val/test.

Speed: ≈ 50 parts/s with 6 workers; the default 20 900 parts take a few minutes on many cores.

**Used by:** [[cli]] (`t2c data synth`), [[data.build]], [[bench.suites]] (`synth`).
**Tested in:** [[test_synth]].
