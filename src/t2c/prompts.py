"""Prompt formats.

Two things are kept separate on purpose:

* the **task text** -- what a benchmark gives every system (the part description);
* the **prompt style** -- how a particular system is asked to answer:

  ``native``    the short system prompt our fine-tuned models are trained with. Output
                contract: a script that leaves the solid in ``result``.
  ``protocol``  the benchmark's own published prompt (used for base and frontier models so
                their numbers are comparable to the literature).
"""

from __future__ import annotations

NATIVE_SYSTEM = """You are an expert CadQuery programmer. Write a Python script that builds the 3D part described by the user.

Rules:
- Start with: import cadquery as cq
- Assign the final solid to a variable named `result`
- Do not export files and do not call show_object(), display() or any other visualization
- Output only the code, inside a single ```python block"""

# Text2CAD-Bench (arXiv 2605.18430, Appendix D.1). The paper prints the role line, four
# critical rules and the start of an API-notes list; its five in-context examples are only
# in supplementary material that is not public. This is therefore the zero-shot core of the
# official prompt, not a byte-exact copy.
T2CB_SYSTEM = """You are an expert CadQuery programmer.

CRITICAL RULES:
1. MUST start with: import cadquery as cq
2. MUST assign final result to variable named 'result'
3. DO NOT use show_object(), display(), or any visualization
4. DO NOT include explanations, ONLY code in ```python``` block

CADQUERY API NOTES:
- There is NO .cone() method.
- Always create geometry before filleting."""

# Text2CAD-Bench Appendix D.2 (error-feedback retry).
RETRY_TEMPLATE = """The previous CadQuery code attempt failed with an error.

## Error Message:
{error_message}

Please analyze the error and generate corrected code.
Output ONLY the corrected code in a ```python``` block"""

# CadQueryEval (github.com/danwahl/cadqueryeval, MIT) -- "default" prompt style, verbatim.
CQE_SYSTEM = """Create valid CadQuery Python code that models the described 3D object
and exports it to 'output.stl'.

IMPORTANT: Create only a SINGLE part/model. Do not include multiple components.

Reminders:
- Ensure the model is manifold and watertight
- Use appropriate CadQuery operations as needed
- Define the model at the origin (0,0,0)
- Do not include example usage or test code
- Do not include explanations outside of code comments

Your code must end with: cq.exporters.export(result, "output.stl")"""

CQE_TASK_TEMPLATE = """## Task Description
{description}

## Target Dimensions
- Bounding box (approximate): {bounding_box_str}
- Expected components: {expected_components}

Generate the CadQuery Python code:"""

SYSTEM_PROMPTS = {"native": NATIVE_SYSTEM, "t2cb": T2CB_SYSTEM, "cqe": CQE_SYSTEM}


def format_completion(code: str) -> str:
    """Target text for training: exactly what we want the model to emit."""
    return f"```python\n{code.strip()}\n```"


def build_messages(task_text: str, system: str = NATIVE_SYSTEM) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": task_text.strip()}]


def retry_messages(messages: list[dict], response: str, error: str) -> list[dict]:
    """Conversation extended with the failed answer and the error-feedback turn."""
    return [
        *messages,
        {"role": "assistant", "content": response},
        {"role": "user", "content": RETRY_TEMPLATE.format(error_message=error or "unknown error")},
    ]
