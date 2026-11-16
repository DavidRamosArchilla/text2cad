"""Turning raw model output into a CadQuery script, and a static pre-flight check.

The contract for every model (base, frontier or fine-tuned) is the one used by
Text2CAD-Bench: a Python script that imports cadquery and leaves the final solid in a
variable called ``result``.
"""

from __future__ import annotations

import ast
import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Modules a CAD script has no business importing. This is a guard against accidents
# (an LLM deciding to "clean up" a directory), not a security boundary -- see sandbox.py.
DENIED_MODULES = {
    "subprocess", "socket", "shutil", "requests", "urllib", "urllib3", "http", "httpx", "ftplib",
    "smtplib", "ctypes", "multiprocessing", "threading", "concurrent", "asyncio", "signal", "pty",
    "importlib", "pickle", "marshal", "builtins", "webbrowser", "resource", "tempfile", "glob",
    "zipfile", "tarfile", "sqlite3", "ssl", "paramiko", "pip", "site", "runpy", "code", "pdb",
}  # fmt: skip
DENIED_CALLS = {"eval", "exec", "compile", "__import__", "input", "breakpoint", "exit", "quit"}
DENIED_ATTRS = {
    "os": {
        "system", "popen", "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace",
        "kill", "killpg", "fork", "forkpty", "chmod", "chown", "truncate", "symlink", "link", "setsid",
        "_exit", "abort", "putenv", "unsetenv", "environ", "getenv", "chroot", "chdir", "walk", "scandir",
    },
    "sys": {"exit", "modules", "setrecursionlimit", "settrace", "setprofile"},
}  # fmt: skip


def strip_thinking(text: str) -> str:
    text = _THINK.sub("", text)
    # An unterminated <think> (generation cut off mid-thought) leaves no usable answer.
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text


def extract_code(text: str) -> str | None:
    """Pull the CadQuery script out of a model response.

    Preference order: the last fenced block tagged python -> the last fenced block of any
    kind -> the raw text if it already looks like a script. Returns None if nothing usable.
    """
    if not text:
        return None
    text = strip_thinking(text)
    blocks = _FENCE.findall(text)
    py = [body for lang, body in blocks if lang.lower() in ("python", "py", "python3")]
    for pool in (py, [body for _, body in blocks]):
        cands = [b for b in pool if b.strip()]
        if cands:
            # Prefer the last block that actually builds something with cadquery.
            for body in reversed(cands):
                if "cadquery" in body or "cq." in body:
                    return body.strip() + "\n"
            return cands[-1].strip() + "\n"
    # Truncated response: an opening fence with no closing one.
    m = re.search(r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*)\Z", text, re.DOTALL)
    if m and ("cadquery" in m.group(1) or "cq." in m.group(1)):
        return m.group(1).strip() + "\n"
    if "import cadquery" in text or "from cadquery" in text:
        start = min(i for i in (text.find("import cadquery"), text.find("from cadquery")) if i >= 0)
        return text[start:].strip() + "\n"
    return None


def static_check(code: str) -> tuple[str, str] | None:
    """Return ``(error_type, message)`` if the script must not be executed, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return "syntax_error", f"SyntaxError: {e.msg} (line {e.lineno})"
    except (ValueError, RecursionError, MemoryError) as e:
        return "syntax_error", f"{type(e).__name__}: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DENIED_MODULES:
                    return "unsafe_code", f"import of '{alias.name}' is not allowed in a CAD script"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in DENIED_MODULES:
                return "unsafe_code", f"import from '{node.module}' is not allowed in a CAD script"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            return "unsafe_code", f"call to '{node.func.id}()' is not allowed in a CAD script"
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in DENIED_ATTRS.get(node.value.id, ()):
                return "unsafe_code", f"use of '{node.value.id}.{node.attr}' is not allowed in a CAD script"
    return None


# --------------------------------------------------------------------------------------
# Normalising third-party training scripts (Gemini/GPT-written DeepCAD translations) to
# the `result` contract. Used by the data preparation pipeline only.
# --------------------------------------------------------------------------------------

_DROP_LINE = re.compile(
    r"^\s*#?\s*(?:"
    r"(?:cq\.)?exporters\.export\(|"
    r"show_object\(|show\(|display\(|debug\(|"
    r"from\s+cadquery\.vis\s+import|from\s+cadquery\s+import\s+exporters\s*$|"
    r"from\s+ocp_vscode\s+import|from\s+jupyter_cadquery|import\s+ocp_vscode|"
    r"-*\s*(?:Export|Show|Display|Visuali[sz]e)\b.*"
    r")",
    re.IGNORECASE,
)
_EXPORT_ARG = re.compile(r"exporters\.export\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,)]")


def _top_level_assignments(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
    return names


def _rename_variable(code: str, tree: ast.Module, old: str, new: str) -> str:
    """Rename a variable by AST position, so attributes, keyword arguments, strings and
    comments that merely contain the same letters are left alone."""
    spots: dict[int, list[tuple[int, int]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == old:
            spots.setdefault(node.lineno, []).append((node.col_offset, node.end_col_offset))
    lines = code.split("\n")
    for lineno, cols in spots.items():
        raw = lines[lineno - 1].encode("utf-8")  # AST column offsets are UTF-8 byte offsets
        for start, end in sorted(cols, reverse=True):
            raw = raw[:start] + new.encode() + raw[end:]
        lines[lineno - 1] = raw.decode("utf-8")
    return "\n".join(lines)


def normalize_reference_script(code: str, final_hint: str | None = None) -> str | None:
    """Strip export/visualisation lines and make the final solid end up in ``result``.

    The final solid is, in order of preference: the object passed to ``exporters.export``,
    ``final_hint`` (a dataset convention such as CAD-Coder's ``r``), an existing ``result``,
    or the last top-level assignment. Returns None when that cannot be done safely.
    """
    exported = _EXPORT_ARG.findall(code)
    kept = [ln.rstrip() for ln in code.replace("\r\n", "\n").split("\n") if not _DROP_LINE.match(ln)]
    body = "\n".join(kept).strip()
    if "import cadquery as cq" not in body:
        return None
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError, RecursionError):
        return None
    assigned = _top_level_assignments(tree)
    if not assigned:
        return None
    for cand in (exported[-1] if exported else None, final_hint, "result", assigned[-1]):
        if cand and cand in assigned:
            final = cand
            break
    else:
        return None
    if final != "result":
        if any(isinstance(n, ast.Name) and n.id == "result" for n in ast.walk(tree)):
            return None  # `result` already means something else in this script
        body = _rename_variable(body, tree, final, "result")
    body = re.sub(r"\n{3,}", "\n\n", body)  # blank-line runs left behind by removed lines
    return body.rstrip() + "\n"
