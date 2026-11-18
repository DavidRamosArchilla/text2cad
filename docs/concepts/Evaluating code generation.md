---
tags: [concept]
aliases: [execution-based evaluation, pass@k]
---
# Evaluating code generation

There are many correct programs for one task, so comparing generated code with a
reference *text* (exact match, BLEU) is a poor measure. The standard is
**execution-based evaluation**: run the code and check what it does. For general code that
means unit tests (HumanEval); for CAD it means building the solid and comparing it with
the correct one — what the verifier in this repository does ([[Walkthrough - scoring one answer]]).

Related ideas:
- **pass@k**: the probability that at least one of *k* sampled answers is correct; pass@1
  with greedy decoding is the usual headline. Our benchmarks report one greedy attempt
  (plus error-feedback retries where the protocol allows), i.e. pass@1-style numbers.
- **Error feedback / self-repair**: give the model its error message and let it retry —
  Text2CAD-Bench allows 3 retries; `IR@1` vs `IR` shows how much they help ([[Metrics]]).
- **Validity is not correctness**: code that runs can still build the wrong part, which is
  why invalid rate alone is a weak metric ([[Text2CAD-Bench]]).

## Learn more
- Chen et al., *Evaluating Large Language Models Trained on Code* (2021; HumanEval and
  pass@k): <https://arxiv.org/abs/2107.03374>
- The CAD benchmarks' own descriptions: [[Text2CAD-Bench]], [[CadQueryEval]], [[CADPrompt]].
