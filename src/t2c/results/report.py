"""Build ``results/REPORT.md`` and ``results/report.html`` from the results registry.

Sections
  1. Scoreboard      one row per model, headline metric of every benchmark, delta vs its base model
  2. Per benchmark   full metric table (ours + published references) and per-group breakdown
  3. Training runs   headline metrics along the checkpoints of each training run
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import yaml

from .. import paths
from ..bench import registry
from . import store

HEADLINE = {  # benchmark -> (metric key, label, higher_is_better, is_percent)
    "t2cb": ("ir", "IR", False, True),
    "cqe": ("pass_rate", "pass", True, True),
    "cadprompt": ("iou_all", "IoU_all", True, False),
    "deepcad": ("iou_all", "IoU_all", True, False),
    "synth": ("iou_all", "IoU_all", True, False),
}
# Extra scoreboard columns: pass/fail is too coarse to see progress on small models.
SCOREBOARD_EXTRA = {"cqe": [("iou_all", "IoU_all", True, False)]}
COLUMNS = [  # key, label, percent, higher_is_better
    ("pass_rate", "Pass ↑", True, True),
    ("ir", "IR ↓", True, False),
    ("ir_first", "IR@1 ↓", True, False),
    ("iou_all", "IoU_all ↑", False, True),
    ("iou", "IoU (valid) ↑", False, True),
    ("cd", "CD×10³ ↓", False, False),
    ("dim_err_median", "dim err ↓", False, False),
]
STAGE_ORDER = {"grpo": 0, "sft": 1, "base": 2, "api": 3}


def _fmt(v, pct=False, nd=3) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.1f}%" if pct else f"{v:.{nd}f}"


def _variant(s: dict) -> str:
    p = s["protocol"]
    return f"{p['prompt_style']}, {p['retries']} retr."


def latest_runs(runs: list[dict]) -> list[dict]:
    """Most recent run per (benchmark, model name, prompt style, retries)."""
    keep: dict[tuple, dict] = {}
    for s in runs:  # sorted by creation time
        keep[(s["benchmark"], s["model"]["name"], s["protocol"]["prompt_style"], s["protocol"]["retries"])] = s
    return list(keep.values())


def _base_run(s: dict, runs: list[dict]) -> dict | None:
    """The base-model run a fine-tuned model should be compared against."""
    base = s["model"].get("base_model")
    if not base or s["model"]["stage"] == "base":
        return None
    cands = [r for r in runs if r["benchmark"] == s["benchmark"] and r["model"]["stage"] == "base" and r["model"]["model"] == base]
    if not cands:
        return None
    # Prefer the same retry budget, then the benchmark's own protocol prompt.
    cands.sort(key=lambda r: (r["protocol"]["retries"] != s["protocol"]["retries"], r["protocol"]["prompt_style"] != "protocol", r["created"]))
    return cands[0]


def _load_references() -> dict:
    p = paths.REPO_ROOT / "configs" / "references.yaml"
    refs = yaml.safe_load(p.read_text()) if p.exists() else {}
    try:
        pub = registry.get("cqe").published()
    except Exception:  # noqa: BLE001
        pub = []
    if pub:
        refs.setdefault("cqe", {"source": "CadQueryEval leaderboard", "caveat": "", "rows": []})
        refs["cqe"]["rows"] = [{"name": r["name"], "group": "overall", "pass_rate": r["pass_rate"], "ir": r["ir"], "date": r["created"]} for r in pub]
    return refs


# ------------------------------------------------------------------------------- tables
class Table:
    def __init__(self, title: str, header: list[str], rows: list[list[str]], note: str = "", ours: list[bool] | None = None):
        self.title, self.header, self.rows, self.note = title, header, rows, note
        self.ours = ours or [True] * len(rows)

    def md(self) -> str:
        out = [f"### {self.title}", ""] if self.title else []
        if self.note:
            out += [f"_{self.note}_", ""]
        out.append("| " + " | ".join(self.header) + " |")
        out.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(self.header))) + "|")
        for row, mine in zip(self.rows, self.ours, strict=True):
            cells = [f"**{c}**" if (mine and i == 0) else c for i, c in enumerate(row)]
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out) + "\n"

    def html(self) -> str:
        e = html.escape
        out = [f"<h3>{e(self.title)}</h3>"] if self.title else []
        if self.note:
            out.append(f'<p class="note">{e(self.note)}</p>')
        out.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{e(h)}</th>" for h in self.header) + "</tr></thead><tbody>")
        for row, mine in zip(self.rows, self.ours, strict=True):
            out.append(f'<tr class="{"ours" if mine else "ref"}">' + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table></div>")
        return "\n".join(out)


def _scoreboard(runs: list[dict], benches: list[str]) -> Table:
    """One row per (model, prompt style); each cell is that benchmark at its own default retry budget."""
    models: dict[tuple, dict] = {}
    for s in runs:  # chronological
        key = (s["model"]["name"], s["protocol"]["prompt_style"])
        slot = models.setdefault(key, {"stage": s["model"]["stage"], "runs": {}})["runs"]
        cur = slot.get(s["benchmark"])
        default = registry.get(s["benchmark"]).default_retries
        if cur is None or s["protocol"]["retries"] == default or cur["protocol"]["retries"] != default:
            slot[s["benchmark"]] = s
    cols = [(b, m) for b in benches for m in [HEADLINE[b], *SCOREBOARD_EXTRA.get(b, [])]]
    header = ["Model", "Stage", "Prompt"] + [f"{b} {m[1]} {'↑' if m[2] else '↓'}" for b, m in cols]
    rows = []
    for (name, style), info in sorted(models.items(), key=lambda kv: (STAGE_ORDER.get(kv[1]["stage"], 9), kv[0])):
        row = [name, info["stage"], style]
        for b, (key, _, _, pct) in cols:
            s = info["runs"].get(b)
            if s is None:
                row.append("–")
                continue
            v = s["metrics"]["overall"].get(key)
            cell = _fmt(v, pct)
            base = _base_run(s, runs)
            if base is not None and v is not None and base["metrics"]["overall"].get(key) is not None:
                d = v - base["metrics"]["overall"][key]
                cell += f" ({'+' if d >= 0 else '−'}{abs(d) * 100:.1f}pp)" if pct else f" ({'+' if d >= 0 else '−'}{abs(d):.3f})"
            row.append(cell)
        rows.append(row)
    budgets = ", ".join(f"{b}: {registry.get(b).default_retries}" for b in benches)
    note = (
        "Headline metric per benchmark, each at its protocol's error-feedback retry budget "
        f"({budgets}). Parentheses: change versus the same model before fine-tuning. "
        "IR = invalid rate; IoU_all counts invalid outputs as 0."
    )
    return Table("", header, rows, note)


def _bench_tables(bench: str, runs: list[dict], refs: dict) -> list[Table]:
    mine = [s for s in runs if s["benchmark"] == bench]
    key, _, hib, _ = HEADLINE[bench]
    cols = [c for c in COLUMNS if any(s["metrics"]["overall"].get(c[0]) is not None for s in mine)]
    ref = refs.get(bench) or {}
    ref_rows = [r for r in ref.get("rows", []) if r.get("group", "overall") == "overall"]
    cols = cols or [c for c in COLUMNS if any(r.get(c[0]) is not None for r in ref_rows)]

    entries = []
    for s in mine:
        m = s["metrics"]["overall"]
        entries.append((m.get(key), True, [s["model"]["name"], s["model"]["stage"], _variant(s)] + [_fmt(m.get(c[0]), c[2]) for c in cols] + [str(m.get("n", "")), s["created"][:10]]))
    shown_refs = ref_rows
    if len(ref_rows) > 14:  # long public leaderboards: the top, plus relatives of what we train
        ranked = sorted(ref_rows, key=lambda r: (r.get(key) is None, -(r.get(key) or 0) if hib else (r.get(key) or 0)))
        shown_refs = ranked[:10] + [r for r in ranked[10:] if "qwen" in r["name"].lower()]
    for r in shown_refs:
        entries.append((r.get(key), False, [r["name"], "published", "see note"] + [_fmt(r.get(c[0]), c[2], 2 if c[0] == "cd" else 3) for c in cols] + ["", str(r.get("date", ""))]))
    entries.sort(key=lambda e: (e[0] is None, -(e[0] or 0) if hib else (e[0] or 0)))
    note = ""
    if ref_rows:
        note = f"Rows marked 'published' — {ref.get('source', '')}. {ref.get('caveat', '')}".strip()
        if len(shown_refs) < len(ref_rows):
            note += f" Showing the top 10 and the Qwen-family entries of {len(ref_rows)} published models."
    tables = [Table("Leaderboard", ["Model", "Stage", "Protocol"] + [c[1] for c in cols] + ["n", "Date"], [e[2] for e in entries], note, [e[1] for e in entries])]

    groups = sorted({g for s in mine for g in s["metrics"]["groups"]})
    if len(groups) > 1 and mine:
        _, label, _, pct = HEADLINE[bench]
        rows, ours = [], []
        for s in sorted(mine, key=lambda s: (STAGE_ORDER.get(s["model"]["stage"], 9), s["model"]["name"])):
            rows.append([s["model"]["name"], _variant(s)] + [_fmt(s["metrics"]["groups"].get(g, {}).get(key), pct) for g in groups])
            ours.append(True)
        by_name: dict[str, dict] = {}
        for r in ref.get("rows", []):
            if r.get("group") in groups:
                by_name.setdefault(r["name"], {})[r["group"]] = r.get(key)
        for name, vals in by_name.items():
            rows.append([name, "published"] + [_fmt(vals.get(g), pct) for g in groups])
            ours.append(False)
        tables.append(Table(f"By group — {label}", ["Model", "Protocol"] + groups, rows, "", ours))

    if mine:
        kinds = sorted({k for s in mine for k in s["errors"]["final"]} - {"ok"})
        if kinds:
            rows = []
            for s in sorted(mine, key=lambda s: (STAGE_ORDER.get(s["model"]["stage"], 9), s["model"]["name"])):
                n = max(sum(s["errors"]["final"].values()), 1)
                rows.append([s["model"]["name"], _variant(s)] + [_fmt(s["errors"]["final"].get(k, 0) / n, True) for k in kinds])
            tables.append(Table("Why outputs were invalid (after retries)", ["Model", "Protocol"] + kinds, rows))
    return tables


def _progress_tables(all_runs: list[dict], benches: list[str]) -> list[Table]:
    tables = []
    by_train: dict[str, list[dict]] = {}
    for s in all_runs:
        if s["model"].get("train_run") and s["model"].get("step") is not None:
            by_train.setdefault(s["model"]["train_run"], []).append(s)
    for train_run, items in sorted(by_train.items()):
        steps = sorted({s["model"]["step"] for s in items})
        if len(steps) < 2:
            continue
        present = [b for b in benches if any(s["benchmark"] == b for s in items)]
        rows = []
        for step in steps:
            row = [str(step)]
            for b in present:
                cand = [s for s in items if s["benchmark"] == b and s["model"]["step"] == step]
                key, _, _, pct = HEADLINE[b]
                row.append(_fmt(cand[-1]["metrics"]["overall"].get(key), pct) if cand else "–")
            rows.append(row)
        stage = items[0]["model"]["stage"]
        tables.append(Table(f"{train_run} ({stage}, base: {items[0]['model'].get('base_model')})", ["Step"] + [f"{b} {HEADLINE[b][1]}" for b in present], rows))
    return tables


_CSS = """
:root{--bg:#fbfaf7;--fg:#1c1b19;--muted:#6b6862;--line:#e3e0d8;--ours:#fff;--ref:#f3f1ea;--accent:#b4532a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171614;--fg:#ece9e2;--muted:#9c978c;--line:#34322d;--ours:#201f1c;--ref:#1a1917;--accent:#e08a5c}}
:root[data-theme="dark"]{--bg:#171614;--fg:#ece9e2;--muted:#9c978c;--line:#34322d;--ours:#201f1c;--ref:#1a1917;--accent:#e08a5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:44px 0 6px;padding-top:18px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:22px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
p.sub,p.note{color:var(--muted);margin:2px 0 10px;max-width:90ch}p.note{font-size:13px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:7px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--muted);font-size:12.5px;cursor:pointer;user-select:none}tr:last-child td{border-bottom:0}
tr.ours td{background:var(--ours)}tr.ours td:first-child{font-weight:600}tr.ref td{background:var(--ref);color:var(--muted)}
tr.ours td:first-child::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);margin-right:8px}
"""
_JS = """
document.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th),
b=t.tBodies[0],d=th.dataset.d=th.dataset.d==='1'?'-1':'1',num=s=>{const m=s.replace('−','-').match(/-?[\\d.]+/);return m?parseFloat(m[0]):NaN};
[...b.rows].sort((x,y)=>{const a=x.cells[i].textContent,c=y.cells[i].textContent,p=num(a),q=num(c);
return (isNaN(p)||isNaN(q)?a.localeCompare(c):p-q)*d}).forEach(r=>b.appendChild(r))}));
"""


def build(open_html: bool = False) -> list[Path]:
    all_runs = store.list_runs()
    runs = latest_runs(all_runs)
    refs = _load_references()
    benches = [b for b in registry.names() if any(s["benchmark"] == b for s in runs) or (refs.get(b) or {}).get("rows")]
    stamp = time.strftime("%Y-%m-%d %H:%M")

    md = ["# Text → CadQuery benchmark report", "", f"_Generated {stamp} from {len(all_runs)} recorded runs. Rebuild with `t2c report`._", ""]
    body = ["<h1>Text → CadQuery benchmark report</h1>", f'<p class="sub">Generated {stamp} from {len(all_runs)} recorded runs. Dotted rows are ours; grey rows are published numbers for context.</p>']

    def section(title: str, intro: str, tables: list[Table]) -> None:
        md.extend([f"## {title}", "", intro, ""] if intro else [f"## {title}", ""])
        body.append(f"<h2>{html.escape(title)}</h2>" + (f'<p class="sub">{html.escape(intro)}</p>' if intro else ""))
        for t in tables:
            md.append(t.md())
            body.append(t.html())

    if runs:
        section("Scoreboard", "", [_scoreboard(runs, [b for b in benches if any(s["benchmark"] == b for s in runs)])])
    else:
        md.append("No evaluation runs recorded yet. Run `t2c eval --model <model>`.\n")
        body.append('<p class="sub">No evaluation runs recorded yet. Run <code>t2c eval --model &lt;model&gt;</code>.</p>')
    for b in benches:
        section(f"{b}", registry.get(b).description, _bench_tables(b, runs, refs))
    prog = _progress_tables(all_runs, benches)
    if prog:
        section("Training runs", "Headline metric at each evaluated checkpoint.", prog)

    out_dir = paths.ensure(paths.results_dir())
    md_path, html_path = out_dir / "REPORT.md", out_dir / "report.html"
    md_path.write_text("\n".join(md))
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Text2CAD benchmark report</title><style>{_CSS}</style></head><body><main>" + "\n".join(body) + f"</main><script>{_JS}</script></body></html>"
    )
    return [md_path, html_path]
