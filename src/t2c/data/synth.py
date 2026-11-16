"""Procedural generator of millimetre-scale mechanical parts with exact ground truth.

The downloadable corpora are DeepCAD sketch-extrude parts in normalised units; the target
benchmarks (Text2CAD-Bench L1/L2, CadQueryEval) describe real parts in millimetres, either
geometrically ("geo") or as a modelling recipe ("seq"). Every part made here comes with

    code   an idiomatic CadQuery script (the training target / ground truth),
    geo    a description of the finished shape,
    seq    step-by-step modelling instructions,

and both texts state every dimension, the axis each one runs along and where the origin
is -- scoring is IoU after bounding-box normalisation with no rotation alignment, so
orientation must be recoverable from the text.

A family is a function ``(rng) -> Part``. It may raise ``Reject`` when the sampled
dimensions do not fit together; ``make_part`` then resamples. Every generated script is
executed in the sandbox and kept only if it yields one watertight solid whose bounding
box (and volume, where known in closed form) match what the generator intended.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .. import paths

PI = math.pi
SPLIT_LETTER = {"train": "t", "val": "v", "test": "x"}  # uid prefix: x = held-out test


class Reject(Exception):
    """Sampled parameters are geometrically infeasible."""


@dataclass
class Part:
    family: str
    code: str
    geo: str
    seq: str
    bbox: tuple[float, float, float]  # intended extents along X, Y, Z
    volume: float | None = None  # closed-form volume when available (consistency check)
    dims: dict = field(default_factory=dict)  # key dimensions both prompts must state ("d_*" may be given as a radius)


# ----------------------------------------------------------------------------- helpers
def nice(rng, lo, hi, step=1):
    """Random multiple of ``step`` in [lo, hi]."""
    a, b = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    if b < a:
        raise Reject
    v = rng.randint(a, b) * step
    return int(v) if float(v).is_integer() else round(v, 4)


def num(v) -> str:
    v = round(float(v), 4)
    return str(int(v)) if v == int(v) else f"{v:g}"


def _pts(points) -> str:
    return "[" + ", ".join(f"({num(x)}, {num(y)})" for x, y in points) + "]"


def J(*parts) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


class V:
    """Renders each dimension either as a named variable or as an inline literal."""

    def __init__(self, rng):
        self.named = rng.random() < 0.6
        self.defs: list[str] = []
        self._seen: set[str] = set()

    def __call__(self, name: str, value) -> str:
        if not self.named:
            return num(value)
        if name not in self._seen:
            self._seen.add(name)
            self.defs.append(f"{name} = {num(value)}")
        return name

    def e(self, expr: str, value) -> str:
        """A derived quantity: the expression when variables are named, else its value."""
        return expr if self.named else num(value)


class Txt:
    """Per-prompt phrasing choices (units, number format, radius vs diameter)."""

    def __init__(self, rng):
        self.r = rng
        self.unit = rng.choice((" mm", " mm", "mm", " millimeters"))
        self.dec = rng.random() < 0.25
        self.radius = rng.random() < 0.25

    def n(self, v) -> str:
        v = round(float(v), 4)
        if v == int(v):
            return f"{v:.1f}" if self.dec else str(int(v))
        return f"{v:g}"

    def mm(self, v) -> str:
        return self.n(v) + self.unit

    def dia(self, d, a=True) -> str:
        s = f"radius of {self.mm(d / 2)}" if self.radius else f"diameter of {self.mm(d)}"
        return ("a " + s) if a else s

    def o(self, *opts) -> str:
        return self.r.choice(opts)


def _script(rng, v: V, lines: list[str], use_math: bool = False) -> str:
    if rng.random() < 0.2:
        lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    head = "import cadquery as cq\n" + ("import math\n" if use_math else "") + "\n"
    if v.defs:
        head += ("# Dimensions (mm)\n" if rng.random() < 0.7 else "") + "\n".join(v.defs) + "\n\n"
    return head + "\n".join(lines) + "\n"


def _geo(t: Txt, *body, plate=False) -> str:
    intro = t.o(
        "The Z-axis is the main axis/thickness direction; the X-axis is the length direction; the Y-axis is the width direction." if plate else "",
        "All dimensions are in millimeters; X, Y and Z are the global coordinate axes.",
        "",
    )
    return J(intro, *body)


def _seq(t: Txt, *body) -> str:
    intro = t.o(
        "First, create a new part file and set the units to millimeters (mm). Establish a global coordinate system: use the origin (0,0,0) "
        "as the reference, with the default XY plane as the base sketch plane and the Z-axis pointing upwards.",
        "Start a new part in millimeters, with the global origin at (0,0,0) and the Z-axis pointing up.",
        "",
    )
    return J(intro, *body)


def _block(rng, v: V, L, W, H, hname="height"):
    """Code for an L x W x H block centred in X/Y. Returns (expression, z of the bottom face)."""
    lx, wy, hz = v("length", L), v("width", W), v(hname, H)
    k = rng.randrange(3)
    if k == 0:
        return f'cq.Workplane("XY").box({lx}, {wy}, {hz})', -H / 2
    if k == 1:
        return f'cq.Workplane("XY").box({lx}, {wy}, {hz}, centered=(True, True, False))', 0
    return f'cq.Workplane("XY").rect({lx}, {wy}).extrude({hz})', 0


def _block_geo(t: Txt, noun, L, W, H, z0, hword="thick") -> str:
    size = t.o(
        f"{t.mm(L)} long (X direction), {t.mm(W)} wide (Y direction) and {t.mm(H)} {hword} (Z direction)",
        f"measuring {t.mm(L)} along X, {t.mm(W)} along Y and {t.mm(H)} along Z",
    )
    place = (
        t.o("It is centered on the origin in X and Y, with its bottom face on the XY plane (Z = 0).", "Its bottom face lies on the XY plane and it is centered on the Z-axis.")
        if z0 == 0
        else t.o("It is centered on the origin in all three directions.", f"The origin is at the center of the body (Z runs from -{t.n(H / 2)} to +{t.n(H / 2)}).")
    )
    return f"{t.o('This is', 'The part is', 'The model is')} {noun} {size}. {place}"


def _block_seq(t: Txt, L, W, H, z0) -> str:
    if z0 == 0:
        return t.o(
            f"On the XY plane, sketch a rectangle {t.mm(L)} (X) by {t.mm(W)} (Y) centered at the origin and extrude it {t.mm(H)} in the +Z direction.",
            f"Draw a {t.n(L)} x {t.n(W)}{t.unit} rectangle centered on the origin in the XY plane (long side along X, short side along Y), then extrude {t.mm(H)} upward along +Z.",
        )
    return t.o(
        f"Create a box {t.mm(L)} along X, {t.mm(W)} along Y and {t.mm(H)} along Z, centered at the origin in all three directions.",
        f"On the XY plane, sketch a rectangle {t.mm(L)} (X) by {t.mm(W)} (Y) centered at the origin and extrude it symmetrically about the plane to a total thickness of {t.mm(H)} (Z from -{t.n(H / 2)} to +{t.n(H / 2)}).",
    )


def _cut_holes(rng, v: V, d, dname="hole_d") -> str:
    if rng.random() < 0.6:
        return f".hole({v(dname, d)})"
    return f".circle({v.e(f'{v(dname, d)} / 2', d / 2)}).cutThruAll()"


# ----------------------------------------------------------------------------- families
def plate_holes(rng) -> Part:
    L = nice(rng, 40, 160, 5)
    W, T, d = nice(rng, 20, min(L, 100), 5), nice(rng, 3, 16), nice(rng, 3, 12)
    layout = rng.choice(("grid", "grid", "corners", "line", "center"))
    edge = rng.choice(("none", "none", "fillet", "chamfer"))
    r = nice(rng, 2, 10) if edge == "fillet" else 0
    c = rng.choice((0.5, 1, 1.5, 2)) if edge == "chamfer" else 0
    nx = ny = 1
    sx = sy = e = 0
    if layout == "grid":
        nx, ny = rng.randint(2, 5), rng.randint(1, 3)
        sx, sy = nice(rng, d + 4, 50, 5), nice(rng, d + 4, 50, 5)
        ex, ey = L / 2 - (nx - 1) * sx / 2, W / 2 - (ny - 1) * sy / 2
        pts = [((i - (nx - 1) / 2) * sx, (j - (ny - 1) / 2) * sy) for i in range(nx) for j in range(ny)]
    elif layout == "corners":
        e = nice(rng, d / 2 + 3, min(L, W) / 2 - d / 2 - 2)
        ex = ey = e
        pts = [(a * (L / 2 - e), b * (W / 2 - e)) for a in (-1, 1) for b in (-1, 1)]
    elif layout == "line":
        e = nice(rng, d / 2 + 3, L / 2 - d / 2 - 2)
        ex, ey, pts = e, W / 2, [(-(L / 2 - e), 0), (L / 2 - e, 0)]
    else:
        d = nice(rng, 6, min(L, W) - 10, 2)
        ex, ey, pts = L / 2, W / 2, [(0, 0)]
    if c >= T / 2 or r > W / 2 - 2 or min(ex, ey) < d / 2 + 2 + c or (r and max(ex, ey) - d / 2 < r + 1):
        raise Reject

    v = V(rng)
    base, z0 = _block(rng, v, L, W, T, "thickness")
    lines = ["# Base plate", f"result = {base}"]
    if r:
        lines += ["# Round the four vertical corners", f'result = result.edges("|Z").fillet({v("corner_r", r)})']
    if c:
        lines += ["# Chamfer the top outer edges", f'result = result.faces(">Z").edges().chamfer({v("chamfer", c)})']
    cut = _cut_holes(rng, v, d)
    if layout == "grid" and ny > 1:
        where = f".rarray({v('pitch_x', sx)}, {v('pitch_y', sy)}, {nx}, {ny})"
    elif layout == "center":
        where = ""
    else:
        where = f".pushPoints({_pts(pts)})"
    lines += ["# Through holes", f'result = result.faces(">Z").workplane(){where}{cut}']
    code = _script(rng, v, lines)

    def holes(t: Txt, seq: bool) -> str:
        thru = t.o("through-holes", "through holes")
        if layout == "grid" and ny > 1:
            s = f"a {nx} x {ny} rectangular pattern of {thru} with {t.dia(d)}: {nx} columns spaced {t.mm(sx)} apart along X and {ny} rows spaced {t.mm(sy)} apart along Y (center to center), the pattern centered on the plate"
        elif layout == "grid":
            s = f"a row of {nx} {thru} with {t.dia(d)} on the centerline parallel to X (Y = 0), centered on the plate, with a center-to-center spacing of {t.mm(sx)}"
        elif layout == "corners":
            s = f"four {thru} with {t.dia(d)}, one near each corner, each centered {t.mm(e)} from the two adjacent edges"
        elif layout == "line":
            s = f"two {thru} with {t.dia(d)} on the length centerline (Y = 0), each centered {t.mm(e)} from the nearer short end"
        else:
            s = f"a single through-hole with {t.dia(d)} at the center of the plate"
        if seq:
            return f"On the top face, sketch {s}; cut {'it' if len(pts) == 1 else 'them'} through the entire thickness."
        return f"The plate has {s}. {t.o('All holes run', 'The holes pass')} through the full thickness along Z."

    tg, ts = Txt(rng), Txt(rng)
    geo = _geo(
        tg,
        _block_geo(tg, "a rectangular plate", L, W, T, z0),
        f"The four vertical corner edges are rounded with a radius of {tg.mm(r)}." if r else "",
        f"The outer edges of the top face are chamfered at 45 degrees with a chamfer width of {tg.mm(c)}." if c else "",
        holes(tg, False),
        plate=True,
    )
    seq = _seq(
        ts,
        _block_seq(ts, L, W, T, z0),
        f"Fillet the four vertical corner edges with a radius of {ts.mm(r)}." if r else "",
        f"Apply a 45-degree chamfer of {ts.mm(c)} to the outer edges of the top face." if c else "",
        holes(ts, True),
    )
    vol = L * W * T - (4 - PI) * r * r * T - (c * c * (L + W) - 4 * c**3 / 3) - len(pts) * PI * d * d / 4 * T
    dims = {"L": L, "W": W, "T": T, "d_hole": d} | ({"r": r} if r else {}) | ({"c": c} if c else {}) | ({"sx": sx} if sx else {}) | ({"e": e} if e else {})
    return Part("plate_holes", code, geo, seq, (L, W, T), vol, dims)


def flange(rng) -> Part:
    OD, T = nice(rng, 50, 160, 10), nice(rng, 6, 20)
    bore = nice(rng, 10, OD * 0.4, 2)
    n, dh = rng.choice((3, 4, 4, 5, 6, 6, 8)), nice(rng, 4, 12)
    hub = rng.random() < 0.5
    hd = nice(rng, bore + 8, OD * 0.6, 2) if hub else 0
    hh = nice(rng, 8, 40, 2) if hub else 0
    pcd = nice(rng, (hd if hub else bore) + dh + 6, OD - dh - 6, 5)
    if pcd * math.sin(PI / n) < dh + 3:
        raise Reject
    v = V(rng)
    od, t, pc, dhv = v("flange_d", OD), v("flange_t", T), v("bolt_circle_d", pcd), v("bolt_d", dh)
    lines = ["# Flange disc", f'result = cq.Workplane("XY").circle({v.e(f"{od} / 2", OD / 2)}).extrude({t})']
    use_math = rng.random() < 0.4
    if use_math:
        lines += [
            f"# {n} bolt holes on the pitch circle",
            f"pts = [({v.e(f'{pc} / 2', pcd / 2)} * math.cos(math.radians(i * 360 / {n})), {v.e(f'{pc} / 2', pcd / 2)} * math.sin(math.radians(i * 360 / {n}))) for i in range({n})]",
            f'result = result.faces(">Z").workplane().pushPoints(pts).circle({v.e(f"{dhv} / 2", dh / 2)}).cutThruAll()',
        ]
    else:
        lines += [f"# {n} bolt holes on the pitch circle", f'result = result.faces(">Z").workplane().polarArray({v.e(f"{pc} / 2", pcd / 2)}, 0, 360, {n}).hole({dhv})']
    if hub:
        lines += ["# Hub", f'result = result.faces(">Z").workplane().circle({v.e(f"{v("hub_d", hd)} / 2", hd / 2)}).extrude({v("hub_h", hh)})']
    lines += ["# Central bore", f'result = result.faces(">Z").workplane().hole({v("bore_d", bore)})']
    code = _script(rng, v, lines, use_math)
    tg, ts = Txt(rng), Txt(rng)
    geo = _geo(
        tg,
        f"{tg.o('This is a circular flange', 'The part is a round mounting flange')} whose axis is the Z-axis. The disc has {tg.dia(OD).replace('a ', 'an outer ', 1)} and a thickness of {tg.mm(T)}; "
        "its bottom face lies on the XY plane (Z = 0) and it is centered on the Z-axis.",
        f"A coaxial cylindrical hub with {tg.dia(hd)} rises {tg.mm(hh)} above the top face of the disc, giving an overall height of {tg.mm(T + hh)}." if hub else "",
        f"A central bore with {tg.dia(bore)} passes through the entire part along the axis.",
        f"{n} bolt holes with {tg.dia(dh)} are equally spaced on a bolt circle (pitch circle) with {tg.dia(pcd)} and go through the disc; one of them lies on the +X axis.",
    )
    seq = _seq(
        ts,
        f"On the XY plane, draw a circle with {ts.dia(OD)} centered at the origin and extrude it {ts.mm(T)} in +Z.",
        f"On the top face, place {n} circles with {ts.dia(dh)} evenly spaced ({num(360 / n)} degrees apart) on a pitch circle with {ts.dia(pcd)}, the first one on the +X axis, and cut them through the disc.",
        f"On the top face, sketch a concentric circle with {ts.dia(hd)} and extrude it {ts.mm(hh)} upward to form the hub." if hub else "",
        f"Finally, cut a concentric hole with {ts.dia(bore)} through the whole part.",
    )
    vol = PI / 4 * (OD**2 * T + hd**2 * hh - bore**2 * (T + hh) - n * dh**2 * T)
    dims = {"d_od": OD, "T": T, "d_bore": bore, "d_bolt": dh, "d_pcd": pcd} | ({"d_hub": hd, "hub_h": hh} if hub else {})
    return Part("flange", code, geo, seq, (OD, OD, T + hh), vol, dims)


def stepped_shaft(rng) -> Part:
    k = rng.choice((2, 3, 3, 4))
    ds = sorted(rng.sample(range(10, 82, 2), k), reverse=True)
    if k >= 3 and rng.random() < 0.4:  # collar in the middle instead of a monotonic taper
        ds = [ds[1], ds[0], *ds[2:]]
    ls = [nice(rng, 5, 60, 5) for _ in range(k)]
    ax = rng.choice("ZZX")
    bore = nice(rng, 4, min(ds) - 6, 2) if rng.random() < 0.35 and min(ds) >= 12 else 0
    c = rng.choice((0.5, 1, 1.5, 2)) if rng.random() < 0.35 else 0
    if c and (c > min(ds[0], ds[-1]) / 2 - bore / 2 - 1.5 or c >= min(ls[0], ls[-1])):
        raise Reject
    total = sum(ls)
    v = V(rng)
    dn = [v(f"d{i + 1}", d) for i, d in enumerate(ds)]
    ln = [v(f"l{i + 1}", le) for i, le in enumerate(ls)]
    plane, top, bot = ('"XY"', ">Z", "<Z") if ax == "Z" else ('"YZ"', ">X", "<X")
    revolve = rng.random() < 0.35
    if revolve:
        prof, z = [(0, 0)], 0
        for d, le in zip(ds, ls, strict=True):
            prof += [(d / 2, z), (d / 2, z + le)]
            z += le
        prof.append((0, z))
        if ax == "Z":
            lines = ["# Half profile (radius, z) revolved about the Z-axis", f'result = cq.Workplane("XZ").polyline({_pts(prof)}).close().revolve(360, (0, 0, 0), (0, 1, 0))']
        else:
            lines = ["# Half profile (x, radius) revolved about the X-axis", f'result = cq.Workplane("XY").polyline({_pts([(b, a) for a, b in prof])}).close().revolve(360, (0, 0, 0), (1, 0, 0))']
        v.defs.clear()
    else:
        lines = ["# Section 1", f"result = cq.Workplane({plane}).circle({v.e(f'{dn[0]} / 2', ds[0] / 2)}).extrude({ln[0]})"]
        for i in range(1, k):
            lines += [f"# Section {i + 1}", f'result = result.faces("{top}").workplane().circle({v.e(f"{dn[i]} / 2", ds[i] / 2)}).extrude({ln[i]})']
    if c:
        lines += ["# Chamfer both ends", f'result = result.faces("{bot}").edges().chamfer({num(c)})', f'result = result.faces("{top}").edges().chamfer({num(c)})']
    if bore:
        lines += ["# Axial through bore", f'result = result.faces("{top}").workplane().hole({num(bore)})']
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)
    start = "the XY plane (Z = 0)" if ax == "Z" else "the YZ plane (X = 0)"
    sections = "; ".join(f"section {i + 1} has {tg.dia(d)} and a length of {tg.mm(le)}" for i, (d, le) in enumerate(zip(ds, ls, strict=True)))
    geo = _geo(
        tg,
        f"{tg.o('This is a stepped shaft', 'The part is a stepped cylindrical shaft')} made of {k} coaxial cylindrical sections along the {ax}-axis. It starts on {start} and extends in the +{ax} direction.",
        f"Listed from the {ax} = 0 end: {sections}. The total length is {tg.mm(total)}.",
        f"A through-bore with {tg.dia(bore)} runs along the axis over the full length." if bore else "",
        f"The outer circular edge of each of the two end faces has a 45-degree chamfer of {tg.mm(c)}." if c else "",
    )
    steps = [f"On {start}, sketch a circle with {ts.dia(ds[0])} centered at the origin and extrude it {ts.mm(ls[0])} in the +{ax} direction."]
    steps += [f"On the new end face, sketch a concentric circle with {ts.dia(d)} and extrude it {ts.mm(le)} further along +{ax}." for d, le in zip(ds[1:], ls[1:], strict=True)]
    seq = _seq(
        ts,
        *steps,
        f"Chamfer the outer circular edge of both end faces by {ts.mm(c)} at 45 degrees." if c else "",
        f"Cut a concentric through-hole with {ts.dia(bore)} along the full length of {ts.mm(total)}." if bore else "",
    )
    vol = sum(PI / 4 * d * d * le for d, le in zip(ds, ls, strict=True)) - PI / 4 * bore**2 * total
    vol -= sum(PI * c * c * (d / 2 - c / 3) for d in (ds[0], ds[-1])) if c else 0
    dm = max(ds)
    bbox = (dm, dm, total) if ax == "Z" else (total, dm, dm)
    dims = {f"d_{i}": d for i, d in enumerate(ds)} | {f"l{i}": le for i, le in enumerate(ls)} | ({"d_bore": bore} if bore else {}) | ({"c": c} if c else {})
    return Part("stepped_shaft", code, geo, seq, bbox, vol, dims)


_POLY = {3: "triangular", 5: "pentagonal", 6: "hexagonal", 8: "octagonal"}


def polygon_prism(rng) -> Part:
    n = rng.choice((3, 5, 6, 6, 6, 8))
    flats = rng.random() < 0.5
    S, H = nice(rng, 16, 80, 2), nice(rng, 4, 50)
    R = S / (2 * math.cos(PI / n)) if flats else S / 2  # circumradius
    ap = R * math.cos(PI / n)  # apothem
    bore = nice(rng, 4, 1.1 * ap, 1) if rng.random() < 0.6 else 0
    c = rng.choice((1, 1.5, 2)) if rng.random() < 0.3 else 0
    if c and (c > H / 3 or c > ap - bore / 2 - 2):
        raise Reject
    v = V(rng)
    size = v("across_flats" if flats else "circum_d", S)
    # CadQuery puts a vertex on +X for polygon(n, d) and the midpoint of a side on +X when circumscribed=True.
    orient = "with the midpoint of one side on the +X axis (that side is perpendicular to X)" if flats else "with one vertex on the +X axis"
    lines = [f"# Regular {n}-gon {orient.split(' (')[0]}", f'result = cq.Workplane("XY").polygon({n}, {size}{", circumscribed=True" if flats else ""}).extrude({v("height", H)})']
    if c:
        lines += ["# Chamfer the top outer edges", f'result = result.faces(">Z").edges().chamfer({num(c)})']
    if bore:
        lines += ["# Central through hole", f'result = result.faces(">Z").workplane(){_cut_holes(rng, v, bore, "bore_d")}']
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)

    def sz(t: Txt) -> str:
        if flats:
            what = "the distance across flats (between opposite sides)" if n % 2 == 0 else "the diameter of the inscribed circle (tangent to the sides)"
            return f"{what} is {t.mm(S)}"
        return f"the circumscribed circle (through the vertices) has a diameter of {t.mm(S)}"

    geo = _geo(
        tg,
        f"{tg.o('This is', 'The model is')} a regular {_POLY[n]} prism ({n} sides) whose axis is the Z-axis. The polygon is centered on the origin {orient}; {sz(tg)}.",
        f"The prism is extruded from the XY plane (Z = 0) to a height of {tg.mm(H)} in +Z.",
        f"The outer edges of the top face are chamfered at 45 degrees by {tg.mm(c)}." if c else "",
        f"A circular through-hole with {tg.dia(bore)} runs along the axis through the full height." if bore else "",
    )
    seq = _seq(
        ts,
        f"On the XY plane, sketch a regular polygon with {n} sides centered at the origin, {orient}; {sz(ts)}. Extrude it {ts.mm(H)} in +Z.",
        f"Apply a 45-degree chamfer of {ts.mm(c)} to the outer edges of the top face." if c else "",
        f"On the top face, sketch a concentric circle with {ts.dia(bore)} and cut it through all." if bore else "",
    )
    area = n / 2 * R * R * math.sin(2 * PI / n)
    vol = area * H - PI / 4 * bore**2 * H - (area * c - area * ap / 3 * (1 - (1 - c / ap) ** 3) if c else 0)
    off = PI / n if flats else 0
    xs = [R * math.cos(off + 2 * PI * i / n) for i in range(n)]
    ys = [R * math.sin(off + 2 * PI * i / n) for i in range(n)]
    dims = {"S": S, "H": H} | ({"d_bore": bore} if bore else {}) | ({"c": c} if c else {})
    return Part("polygon_prism", code, geo, seq, (max(xs) - min(xs), max(ys) - min(ys), H), vol, dims)


def _leg_holes(rng, W, d):
    """One hole on the mid-plane or two holes symmetric about it. Returns (y positions, spacing)."""
    if rng.random() < 0.5:
        return [0], 0
    s = nice(rng, d + 6, W - d - 6, 2)
    return [-s / 2, s / 2], s


def l_bracket(rng) -> Part:
    A, B, W, t = nice(rng, 30, 100, 5), nice(rng, 30, 100, 5), nice(rng, 20, 80, 5), nice(rng, 3, 10)
    d = nice(rng, 4, 10)
    e1, e2 = nice(rng, d / 2 + 3, A - t - d / 2 - 2), nice(rng, d / 2 + 3, B - t - d / 2 - 2)
    y1, s1 = _leg_holes(rng, W, d)
    y2, s2 = _leg_holes(rng, W, d)
    v = V(rng)
    a, b, w, tt, dv = v("leg_x", A), v("leg_z", B), v("width", W), v("thickness", t), v("hole_d", d)
    if rng.random() < 0.5:
        prof = f"[(0, 0), ({a}, 0), ({a}, {tt}), ({tt}, {tt}), ({tt}, {b}), (0, {b})]"
        lines = ["# L-shaped profile in the XZ plane, extruded symmetrically along Y", f'result = cq.Workplane("XZ").polyline({prof}).close().extrude({v.e(f"{w} / 2", W / 2)}, both=True)']
    else:
        lines = [
            "# Horizontal and vertical legs",
            f'base = cq.Workplane("XY").box({a}, {w}, {tt}, centered=(False, True, False))',
            f'wall = cq.Workplane("XY").box({tt}, {w}, {b}, centered=(False, True, False))',
            "result = base.union(wall)",
        ]
    ph, pv = [(A - e1, y) for y in y1], [(y, B - e2) for y in y2]
    if rng.random() < 0.5:
        r2 = v.e(f"{dv} / 2", d / 2)
        lines += [
            "# Holes through the horizontal leg (along Z)",
            f'result = result.cut(cq.Workplane("XY").pushPoints({_pts(ph)}).circle({r2}).extrude({tt}))',
            "# Holes through the vertical leg (along X)",
            f'result = result.cut(cq.Workplane("YZ").pushPoints({_pts(pv)}).circle({r2}).extrude({tt}))',
        ]
    else:
        lines += [
            "# Holes through the horizontal leg (along Z)",
            f'result = result.faces("<Z").workplane().pushPoints({_pts(ph)}).hole({dv})',
            "# Holes through the vertical leg (along X)",
            f'result = result.faces("<X").workplane().pushPoints({_pts(pv)}).hole({dv})',
        ]
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)

    def hole_txt(t: Txt, leg: str, axis: str, e, edge: str, ys, s) -> str:
        where = "on the mid-plane Y = 0" if len(ys) == 1 else f"spaced {t.mm(s)} apart along Y, symmetric about the mid-plane Y = 0"
        cnt = "one through-hole" if len(ys) == 1 else "two through-holes"
        return f"The {leg} leg has {cnt} with {t.dia(d)} along {axis}, centered {t.mm(e)} from {edge}, {where}."

    geo = _geo(
        tg,
        f"{tg.o('This is an L-shaped angle bracket', 'The part is a right-angle (L) bracket')}. The horizontal leg lies on the XY plane and extends {tg.mm(A)} along +X, measured from the outer corner; "
        f"the vertical leg rises along +Z to a height of {tg.mm(B)}, measured from the bottom face. Both legs are {tg.mm(t)} thick, and the bracket is {tg.mm(W)} wide along Y, symmetric about the XZ plane.",
        "The outer corner edge of the L lies on the Y-axis (X = 0, Z = 0).",
        hole_txt(tg, "horizontal", "Z", e1, "its free end (the face at X = " + tg.n(A) + ")", y1, s1),
        hole_txt(tg, "vertical", "X", e2, "its top edge (the face at Z = " + tg.n(B) + ")", y2, s2),
    )
    seq = _seq(
        ts,
        f"On the XZ plane, sketch a closed L-shaped profile through the points (X, Z): (0, 0), ({ts.n(A)}, 0), ({ts.n(A)}, {ts.n(t)}), ({ts.n(t)}, {ts.n(t)}), ({ts.n(t)}, {ts.n(B)}), (0, {ts.n(B)}), in{ts.unit.strip() and ' ' + ts.unit.strip()}.",
        f"Extrude the profile symmetrically about the sketch plane, {ts.mm(W / 2)} to each side, for a total width of {ts.mm(W)} along Y.",
        f"Cut {'a hole' if len(y1) == 1 else 'two holes'} with {ts.dia(d)} through the horizontal leg along Z, centered at X = {ts.n(A - e1)}" + (", Y = 0." if len(y1) == 1 else f" and Y = +/-{ts.n(s1 / 2)}."),
        f"Cut {'a hole' if len(y2) == 1 else 'two holes'} with {ts.dia(d)} through the vertical leg along X, centered at Z = {ts.n(B - e2)}" + (", Y = 0." if len(y2) == 1 else f" and Y = +/-{ts.n(s2 / 2)}."),
    )
    vol = (A * t + (B - t) * t) * W - (len(y1) + len(y2)) * PI / 4 * d * d * t
    return Part("l_bracket", code, geo, seq, (A, W, B), vol, {"A": A, "B": B, "W": W, "t": t, "d_hole": d})


def beam_profile(rng) -> Part:
    kind = rng.choice(("U", "T", "I"))
    b, h, tw, tf, L = nice(rng, 20, 80, 5), nice(rng, 20, 100, 5), nice(rng, 2, 8), nice(rng, 2, 8), nice(rng, 40, 200, 10)
    if b - 2 * tw < 6 or h - 2 * tf < 6:
        raise Reject
    if kind == "U":
        pts = [(-b / 2, 0), (b / 2, 0), (b / 2, h), (b / 2 - tw, h), (b / 2 - tw, tf), (-b / 2 + tw, tf), (-b / 2 + tw, h), (-b / 2, h)]
        area, name = b * h - (b - 2 * tw) * (h - tf), "U-channel (open at the top)"
        desc = "The base (web) along the bottom is {tf} thick and the two vertical side walls are {tw} thick each."
    elif kind == "T":
        pts = [(-tw / 2, 0), (tw / 2, 0), (tw / 2, h - tf), (b / 2, h - tf), (b / 2, h), (-b / 2, h), (-b / 2, h - tf), (-tw / 2, h - tf)]
        area, name = b * tf + tw * (h - tf), "T-section beam"
        desc = "The horizontal flange at the top is {tf} thick and spans the full width; the vertical web below it is {tw} thick, centered on the width, and reaches down to Z = 0."
    else:
        pts = [(-b / 2, 0), (b / 2, 0), (b / 2, tf), (tw / 2, tf), (tw / 2, h - tf), (b / 2, h - tf), (b / 2, h), (-b / 2, h), (-b / 2, h - tf), (-tw / 2, h - tf), (-tw / 2, tf), (-b / 2, tf)]
        area, name = 2 * b * tf + tw * (h - 2 * tf), "I-beam (H-section)"
        desc = "The top and bottom horizontal flanges are each {tf} thick and span the full width; they are joined by a central vertical web {tw} thick."
    along_x = rng.random() < 0.5
    v = V(rng)
    if along_x:
        ext, plane, u, ln = f"extrude({v('length', L)})", "YZ", "Y", "from X = 0 to X = {L}"
    else:
        ext, plane, u, ln = f"extrude({v.e(f'{v("length", L)} / 2', L / 2)}, both=True)", "XZ", "X", "symmetrically about the XZ plane, from Y = -{h2} to Y = +{h2}"
    lines = [f"# {name} cross-section in the {plane} plane", f"profile = {_pts(pts)}", f'result = cq.Workplane("{plane}").polyline(profile).close().{ext}']
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)
    la = "X" if along_x else "Y"
    geo = _geo(
        tg,
        f"{tg.o('This is', 'The part is')} a {name} of constant cross-section, {tg.mm(L)} long along the {la}-axis; it extends " + ln.format(L=tg.n(L), h2=tg.n(L / 2)) + ".",
        f"The cross-section lies in the {plane} plane: its overall width is {tg.mm(b)} along {u}, centered on {u} = 0, and its overall height is {tg.mm(h)} along Z, with the bottom at Z = 0.",
        desc.format(tf=tg.mm(tf), tw=tg.mm(tw)),
    )
    seq = _seq(
        ts,
        f"On the {plane} plane, sketch the closed cross-section of a {name} through the points ({u}, Z) = " + ", ".join(f"({ts.n(x)}, {ts.n(z)})" for x, z in pts) + f" (values in mm), i.e. overall width {ts.mm(b)}, overall height {ts.mm(h)}, web thickness {ts.mm(tw)} and flange/base thickness {ts.mm(tf)}.",
        f"Extrude the profile {ts.mm(L)} in the +X direction." if along_x else f"Extrude the profile symmetrically about the sketch plane, {ts.mm(L / 2)} to each side, for a total length of {ts.mm(L)} along Y.",
    )
    bbox = (L, b, h) if along_x else (b, L, h)
    return Part("beam_profile", code, geo, seq, bbox, area * L, {"b": b, "h": h, "tw": tw, "tf": tf, "L": L})


def open_box(rng) -> Part:
    L = nice(rng, 40, 160, 10)
    W, H, t = nice(rng, 30, L, 10), nice(rng, 15, 80, 5), nice(rng, 1.5, 5, 0.5)
    r = nice(rng, t + 2, min(L, W) / 4) if rng.random() < 0.35 else 0
    dh = nice(rng, 4, min(L, W) / 3) if rng.random() < 0.3 else 0
    if H - t < 5:
        raise Reject
    v = V(rng)
    lx, wy, hz, tt = v("length", L), v("width", W), v("height", H), v("wall", t)
    idiom = 0 if r else rng.randrange(3)
    z0 = 0
    if idiom == 0:
        centered = rng.random() < 0.5
        z0 = -H / 2 if centered else 0
        box = f'cq.Workplane("XY").box({lx}, {wy}, {hz}{"" if centered else ", centered=(True, True, False)"})'
        fil = f'.edges("|Z").fillet({v("corner_r", r)})' if r else ""
        lines = ["# Solid block, then remove the top face and hollow it out", f'result = {box}{fil}.faces(">Z").shell(-{tt})']
    elif idiom == 1:
        lines = [
            "# Outer block minus the inner cavity",
            f'outer = cq.Workplane("XY").box({lx}, {wy}, {hz}, centered=(True, True, False))',
            f'cavity = cq.Workplane("XY").workplane(offset={tt}).box({v.e(f"{lx} - 2 * {tt}", L - 2 * t)}, {v.e(f"{wy} - 2 * {tt}", W - 2 * t)}, {v.e(f"{hz} - {tt}", H - t)}, centered=(True, True, False))',
            "result = outer.cut(cavity)",
        ]
    else:
        lines = [
            "# Block with a rectangular pocket cut from the top, leaving the floor",
            f'result = cq.Workplane("XY").rect({lx}, {wy}).extrude({hz})',
            f'result = result.faces(">Z").workplane().rect({v.e(f"{lx} - 2 * {tt}", L - 2 * t)}, {v.e(f"{wy} - 2 * {tt}", W - 2 * t)}).cutBlind(-{v.e(f"({hz} - {tt})", H - t)})',
        ]
    if dh:
        lines += ["# Hole through the floor", f'result = result.faces("<Z").workplane().hole({v("floor_hole_d", dh)})']
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)
    geo = _geo(
        tg,
        _block_geo(tg, tg.o("an open-top rectangular box (tray) with outer dimensions", "a rectangular enclosure body, open at the top, with outer dimensions"), L, W, H, z0, "high"),
        f"The four side walls and the floor all have a uniform thickness of {tg.mm(t)}; the top (+Z side) is completely open, so the inner cavity measures {tg.mm(L - 2 * t)} by {tg.mm(W - 2 * t)} and is {tg.mm(H - t)} deep.",
        f"The four outer vertical corner edges are rounded with a radius of {tg.mm(r)}, and the inner corners follow with a radius of {tg.mm(r - t)} so the wall thickness stays constant." if r else "",
        f"A circular hole with {tg.dia(dh)} passes through the center of the floor along Z." if dh else "",
        plate=True,
    )
    seq = _seq(
        ts,
        _block_seq(ts, L, W, H, z0),
        f"Fillet the four vertical corner edges with a radius of {ts.mm(r)}." if r else "",
        ts.o(
            f"Remove the top face and shell the solid inward with a wall thickness of {ts.mm(t)}, leaving the top open.",
            f"On the top face, sketch a centered rectangle {ts.mm(L - 2 * t)} by {ts.mm(W - 2 * t)} and cut it down to a depth of {ts.mm(H - t)}, leaving walls and a floor {ts.mm(t)} thick.",
        )
        if not r
        else f"Remove the top face and shell the solid inward with a wall thickness of {ts.mm(t)} (inner corner radius {ts.mm(r - t)}), leaving the top open.",
        f"Cut a hole with {ts.dia(dh)} through the center of the floor." if dh else "",
    )
    vol = (L * W - (4 - PI) * r * r) * H - ((L - 2 * t) * (W - 2 * t) - ((4 - PI) * (r - t) ** 2 if r else 0)) * (H - t) - PI / 4 * dh * dh * t
    dims = {"L": L, "W": W, "H": H, "t": t} | ({"r": r} if r else {}) | ({"d_floor": dh} if dh else {})
    return Part("open_box", code, geo, seq, (L, W, H), vol, dims)


def tube_flange(rng) -> Part:
    OD = nice(rng, 12, 60, 2)
    ID, L = nice(rng, 6, OD - 4, 2), nice(rng, 10, 80, 5)
    fl = rng.random() < 0.6
    FD, FT = (nice(rng, OD + 8, OD + 40, 2), nice(rng, 3, 10)) if fl else (0, 0)
    if fl and L - FT < 5:
        raise Reject
    ax = rng.choice("ZZX")
    plane, top = ('"XY"', ">Z") if ax == "Z" else ('"YZ"', ">X")
    v = V(rng)
    od, idv, ln = v("outer_d", OD), v("inner_d", ID), v("length", L)
    ro, ri = v.e(f"{od} / 2", OD / 2), v.e(f"{idv} / 2", ID / 2)
    if not fl:
        if rng.random() < 0.5:
            lines = ["# Annular section extruded along the axis", f"result = cq.Workplane({plane}).circle({ro}).circle({ri}).extrude({ln})"]
        else:
            lines = ["# Solid cylinder, then bore", f"result = cq.Workplane({plane}).circle({ro}).extrude({ln})", f'result = result.faces("{top}").workplane().hole({idv})']
    else:
        fd, ft = v("flange_d", FD), v("flange_t", FT)
        rf, body = v.e(f"{fd} / 2", FD / 2), v.e(f"{ln} - {ft}", L - FT)
        if rng.random() < 0.5:
            lines = [
                "# Flange, body, then the bore through both",
                f"result = cq.Workplane({plane}).circle({rf}).extrude({ft})",
                f'result = result.faces("{top}").workplane().circle({ro}).extrude({body})',
                f'result = result.faces("{top}").workplane().hole({idv})',
            ]
        else:
            lines = [
                "# Flange ring and tube body, both hollow",
                f"result = cq.Workplane({plane}).circle({rf}).circle({ri}).extrude({ft})",
                f'result = result.faces("{top}").workplane().circle({ro}).circle({ri}).extrude({body})',
            ]
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)
    start = "the XY plane (Z = 0)" if ax == "Z" else "the YZ plane (X = 0)"
    geo = _geo(
        tg,
        f"{tg.o('This is', 'The part is')} a {'flanged bushing (a hollow cylinder with a flange at one end)' if fl else 'plain tube (hollow cylinder)'} whose axis is the {ax}-axis. "
        f"It starts on {start} and extends {tg.mm(L)} in the +{ax} direction{' (total length including the flange)' if fl else ''}.",
        f"The tubular body has {tg.dia(OD).replace('a ', 'an outer ', 1)} and a bore with {tg.dia(ID)} that passes through the entire length.",
        f"At the {ax} = 0 end there is a circular flange with {tg.dia(FD)} and a thickness of {tg.mm(FT)}, coaxial with the body." if fl else "",
    )
    seq = _seq(
        ts,
        f"On {start}, sketch a circle with {ts.dia(FD)} centered at the origin and extrude it {ts.mm(FT)} in +{ax} to form the flange. On its end face, sketch a concentric circle with {ts.dia(OD)} and extrude it a further {ts.mm(L - FT)} (overall length {ts.mm(L)})."
        if fl
        else f"On {start}, sketch a circle with {ts.dia(OD)} centered at the origin and extrude it {ts.mm(L)} in the +{ax} direction.",
        f"Cut a concentric through-hole with {ts.dia(ID)} along the whole length.",
    )
    vol = PI / 4 * ((OD**2 - ID**2) * L + ((FD**2 - OD**2) * FT if fl else 0))
    dm = max(OD, FD)
    dims = {"d_od": OD, "d_id": ID, "L": L} | ({"d_flange": FD, "FT": FT} if fl else {})
    return Part("tube_flange", code, geo, seq, (dm, dm, L) if ax == "Z" else (L, dm, dm), vol, dims)


def slotted_block(rng) -> Part:
    L, W, H = nice(rng, 40, 120, 5), nice(rng, 20, 80, 5), nice(rng, 10, 50, 5)
    kind = rng.choice(("groove", "pocket", "obround"))
    v = V(rng)
    base, z0 = _block(rng, v, L, W, H)
    lines = ["# Base block", f"result = {base}"]
    tg, ts = Txt(rng), Txt(rng)
    if kind == "groove":
        along = rng.choice("XY")
        gw, gd = nice(rng, 4, (W if along == "X" else L) - 8, 2), nice(rng, 2, H - 4)
        rect = f"{v('length', L)}, {v('groove_w', gw)}" if along == "X" else f"{v('groove_w', gw)}, {v('width', W)}"
        lines += [f"# Groove along {along} in the top face", f'result = result.faces(">Z").workplane().rect({rect}).cutBlind(-{v("groove_depth", gd)})']
        across = "Y" if along == "X" else "X"
        g = f"a straight rectangular groove (channel) in the top face, {tg.mm(gw)} wide and {tg.mm(gd)} deep, running the full length of the block along {along} and centered on {across} = 0"
        s = f"On the top face, sketch a rectangle {ts.mm(gw)} wide (along {across}), centered, that spans the entire block along {along}; cut it to a depth of {ts.mm(gd)} to form an open-ended groove."
        vol, dims = L * W * H - (L if along == "X" else W) * gw * gd, {"gw": gw, "gd": gd}
    elif kind == "pocket":
        pl, pw, pd = nice(rng, 10, L - 10, 5), nice(rng, 8, W - 8, 2), nice(rng, 2, H - 3)
        pr = nice(rng, 1, min(pl, pw) / 2 - 1) if rng.random() < 0.5 else 0
        if pr:
            lines += [
                "# Pocket cutter with rounded vertical corners",
                f'pocket = cq.Workplane("XY").box({v("pocket_l", pl)}, {v("pocket_w", pw)}, {v("pocket_depth", pd)}).edges("|Z").fillet({v("pocket_r", pr)})',
                f"result = result.cut(pocket.translate((0, 0, {num(z0 + H - pd / 2)})))",
            ]
        else:
            lines += ["# Blind pocket in the top face", f'result = result.faces(">Z").workplane().rect({v("pocket_l", pl)}, {v("pocket_w", pw)}).cutBlind(-{v("pocket_depth", pd)})']
        rr = (lambda t: f", with its four vertical corners rounded to a radius of {t.mm(pr)}") if pr else (lambda t: "")
        g = f"a blind rectangular pocket centered on the top face, {tg.mm(pl)} along X by {tg.mm(pw)} along Y and {tg.mm(pd)} deep{rr(tg)}"
        s = f"On the top face, sketch a centered rectangle {ts.mm(pl)} (X) by {ts.mm(pw)} (Y){rr(ts)}, and cut it to a depth of {ts.mm(pd)}."
        vol, dims = L * W * H - (pl * pw - (4 - PI) * pr * pr) * pd, {"pl": pl, "pw": pw, "pd": pd} | ({"pr": pr} if pr else {})
    else:
        ns, sd = rng.choice((1, 1, 2, 3)), nice(rng, 4, 12)
        sl, p = nice(rng, sd + 6, L - 10, 2), nice(rng, sd + 4, 30, 2)
        if (ns - 1) * p + sd > W - 8:
            raise Reject
        ys = [(i - (ns - 1) / 2) * p for i in range(ns)]
        push = "" if ns == 1 else f".pushPoints({_pts([(0, y) for y in ys])})"
        lines += ["# Obround through slot(s), long axis along X", f'result = result.faces(">Z").workplane(){push}.slot2D({v("slot_len", sl)}, {v("slot_w", sd)}).cutThruAll()']
        many = "" if ns == 1 else ", spaced {p} apart (center to center) along Y and arranged symmetrically about Y = 0"
        cnt = "one obround (stadium-shaped) through-slot" if ns == 1 else f"{ns} identical obround (stadium-shaped) through-slots"
        g = f"{cnt} cut through the full height along Z; each slot is {tg.mm(sl)} long end to end along X and {tg.mm(sd)} wide, with semicircular ends, centered on X = 0{many.format(p=tg.mm(p))}"
        s = f"On the top face, sketch {cnt.replace('through-', '')} with the long axis along X: overall length {ts.mm(sl)}, width {ts.mm(sd)} (end radius {ts.mm(sd / 2)}), centered on X = 0{many.format(p=ts.mm(p))}; cut through all."
        vol, dims = L * W * H - ns * ((sl - sd) * sd + PI / 4 * sd * sd) * H, {"sl": sl, "sd": sd} | ({"p": p} if ns > 1 else {})
    code = _script(rng, v, lines)
    geo = _geo(tg, _block_geo(tg, "a rectangular block", L, W, H, z0, "high"), f"It has {g}.", plate=True)
    seq = _seq(ts, _block_seq(ts, L, W, H, z0), s)
    return Part("slotted_block", code, geo, seq, (L, W, H), vol, {"L": L, "W": W, "H": H} | dims)


def cbore_plate(rng) -> Part:
    L, W, T = nice(rng, 40, 140, 5), nice(rng, 30, 100, 5), nice(rng, 8, 25)
    d = nice(rng, 4, 10)
    D = nice(rng, d + 4, d + 10)
    csk = rng.random() < 0.4
    ang = rng.choice((82, 90)) if csk else 0
    cd = (D - d) / 2 / math.tan(math.radians(ang / 2)) if csk else nice(rng, 2, T - 3)
    if cd > T - 1.5:
        raise Reject
    layout = rng.choice(("corners", "corners", "line"))
    if layout == "corners":
        e = nice(rng, D / 2 + 3, min(L, W) / 2 - D / 2 - 1)
        pts = [(a * (L / 2 - e), b * (W / 2 - e)) for a in (-1, 1) for b in (-1, 1)]
    else:
        nh = rng.choice((2, 3))
        e = nice(rng, D / 2 + 3, L / 2 - D - 2)
        pts = [(-(L / 2 - e) + i * (L - 2 * e) / (nh - 1), 0) for i in range(nh)]
    v = V(rng)
    base, z0 = _block(rng, v, L, W, T, "thickness")
    call = f"cskHole({v('hole_d', d)}, {v('csk_d', D)}, {ang})" if csk else f"cboreHole({v('hole_d', d)}, {v('cbore_d', D)}, {v('cbore_depth', cd)})"
    if layout == "corners" and rng.random() < 0.5:
        where = f".rect({v.e(f'{v("length", L)} - 2 * {num(e)}', L - 2 * e)}, {v.e(f'{v("width", W)} - 2 * {num(e)}', W - 2 * e)}, forConstruction=True).vertices()"
    else:
        where = f".pushPoints({_pts(pts)})"
    lines = ["# Base plate", f"result = {base}", f"# {'Countersunk' if csk else 'Counterbored'} holes from the top face", f'result = result.faces(">Z").workplane(){where}.{call}']
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)

    def feat(t: Txt) -> str:
        if csk:
            return f"countersunk through-holes: a through-hole with {t.dia(d)} and, at the top face, a conical countersink with {t.dia(D)} and an included angle of {ang} degrees"
        return f"counterbored through-holes: a through-hole with {t.dia(d)} and a coaxial counterbore with {t.dia(D)}, {t.mm(cd)} deep from the top face"

    def pos(t: Txt) -> str:
        if layout == "corners":
            return f"four of them, one near each corner, each centered {t.mm(e)} from the two adjacent edges"
        return f"{len(pts)} of them on the length centerline (Y = 0), equally spaced, with the outermost holes centered {t.mm(e)} from the short ends"

    geo = _geo(tg, _block_geo(tg, "a rectangular plate", L, W, T, z0), f"The plate has {feat(tg)}. There are {pos(tg)}. All hole axes are parallel to Z.", plate=True)
    seq = _seq(ts, _block_seq(ts, L, W, T, z0), f"On the top face, create {feat(ts)}; place {pos(ts)}.")
    extra = (PI * cd / 12 * (D * D + D * d + d * d) - PI / 4 * d * d * cd) if csk else PI / 4 * (D * D - d * d) * cd
    vol = L * W * T - len(pts) * (PI / 4 * d * d * T + extra)
    dims = {"L": L, "W": W, "T": T, "d_hole": d, "d_cb": D, "e": e} | ({} if csk else {"cd": cd})
    return Part("cbore_plate", code, geo, seq, (L, W, T), vol, dims)


def revolved(rng) -> Part:
    kind = rng.choice(("frustum", "pulley", "dome"))
    v = V(rng)
    tg, ts = Txt(rng), Txt(rng)
    use_xz = "On the XZ plane, sketch the closed half-profile through the points (radius, Z) = "
    if kind == "frustum":
        D1, D2, H = nice(rng, 20, 100, 5), nice(rng, 10, 100, 5), nice(rng, 10, 80, 5)
        if D1 == D2:
            raise Reject
        b = nice(rng, 4, min(D1, D2) - 6, 2) if rng.random() < 0.4 else 0
        k = rng.randrange(3)
        r1, r2, hh = v.e(f"{v('bottom_d', D1)} / 2", D1 / 2), v.e(f"{v('top_d', D2)} / 2", D2 / 2), v("height", H)
        if k == 0:
            ri = num(b / 2)
            lines = ["# Half profile (radius, z) revolved about the Z-axis", f'result = cq.Workplane("XZ").polyline([({ri}, 0), ({r1}, 0), ({r2}, {hh}), ({ri}, {hh})]).close().revolve(360, (0, 0, 0), (0, 1, 0))']
        else:
            make = f'cq.Workplane("XY").circle({r1}).workplane(offset={hh}).circle({r2}).loft(combine=True)' if k == 1 else f'cq.Workplane("XY").add(cq.Solid.makeCone({r1}, {r2}, {hh}))'
            lines = ["# Truncated cone", f"result = {make}"] + ([f'result = result.faces(">Z").workplane().hole({num(b)})'] if b else [])
        geo = _geo(
            tg,
            f"{tg.o('This is', 'The part is')} a truncated cone (conical frustum) whose axis is the Z-axis. The bottom face lies on the XY plane with {tg.dia(D1)}; the top face is at Z = {tg.n(H)} with {tg.dia(D2)}; the height is {tg.mm(H)}.",
            f"A coaxial through-bore with {tg.dia(b)} runs the full height." if b else "",
        )
        seq = _seq(
            ts,
            use_xz + f"({ts.n(b / 2)}, 0), ({ts.n(D1 / 2)}, 0), ({ts.n(D2 / 2)}, {ts.n(H)}), ({ts.n(b / 2)}, {ts.n(H)}) in mm, and revolve it 360 degrees about the Z-axis."
            + (f" This gives a frustum with base diameter {ts.mm(D1)}, top diameter {ts.mm(D2)}, height {ts.mm(H)} and a bore of diameter {ts.mm(b)}." if b else f" This gives a solid frustum with base diameter {ts.mm(D1)}, top diameter {ts.mm(D2)} and height {ts.mm(H)}."),
        )
        vol = PI * H / 12 * (D1 * D1 + D1 * D2 + D2 * D2) - PI / 4 * b * b * H
        dm, bbox, dims = max(D1, D2), None, {"d_1": D1, "d_2": D2, "H": H} | ({"d_bore": b} if b else {})
        bbox = (dm, dm, H)
    elif kind == "pulley":
        OD, Wd = nice(rng, 40, 120, 5), nice(rng, 10, 30, 2)
        b, gd, gw = nice(rng, 6, OD / 3, 2), nice(rng, 3, 10), nice(rng, 4, Wd - 4, 2)
        gb = nice(rng, 1, gw - 2)
        R = OD / 2
        if R - gd < b / 2 + 4:
            raise Reject
        pts = [(b / 2, 0), (R, 0), (R, (Wd - gw) / 2), (R - gd, (Wd - gb) / 2), (R - gd, (Wd + gb) / 2), (R, (Wd + gw) / 2), (R, Wd), (b / 2, Wd)]
        lines = ["# Half profile (radius, z) with the V-groove, revolved about the Z-axis", f"profile = {_pts(pts)}", 'result = cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))']
        geo = _geo(
            tg,
            f"{tg.o('This is', 'The part is')} a V-groove pulley whose axis is the Z-axis: a disc with {tg.dia(OD).replace('a ', 'an outer ', 1)} and a width of {tg.mm(Wd)} along Z (from Z = 0 to Z = {tg.n(Wd)}), with a central bore with {tg.dia(b)}.",
            f"A circumferential groove is cut into the rim, symmetric about the mid-plane Z = {tg.n(Wd / 2)}: it is {tg.mm(gw)} wide at the outer diameter, narrows linearly to {tg.mm(gb)} at its flat bottom, and is {tg.mm(gd)} deep measured radially.",
        )
        seq = _seq(
            ts,
            use_xz + ", ".join(f"({ts.n(x)}, {ts.n(z)})" for x, z in pts) + " in mm.",
            f"Revolve the profile 360 degrees about the Z-axis to obtain a pulley of outer diameter {ts.mm(OD)}, width {ts.mm(Wd)} and bore diameter {ts.mm(b)}, with a rim groove {ts.mm(gw)} wide at the top, {ts.mm(gb)} wide at the bottom and {ts.mm(gd)} deep.",
        )
        ga, rc = gd * (gw + gb) / 2, R - gd * (gw + 2 * gb) / (3 * (gw + gb))
        vol, bbox = PI / 4 * (OD * OD - b * b) * Wd - 2 * PI * rc * ga, (OD, OD, Wd)
        dims = {"d_od": OD, "Wd": Wd, "d_bore": b, "gd": gd, "gw": gw, "gb": gb}
    else:
        D, Hc = nice(rng, 10, 60, 2), nice(rng, 10, 80, 5)
        R = D / 2
        if Hc >= R and rng.random() < 0.5:
            lines = ["# Cylinder with a sphere centred on its top face", f'result = cq.Workplane("XY").circle({num(R)}).extrude({num(Hc)}).union(cq.Workplane("XY").sphere({num(R)}).translate((0, 0, {num(Hc)})))']
        else:
            m = R * math.sqrt(0.5)
            lines = [
                "# Half profile with a quarter-circle arc, revolved about the Z-axis",
                f'result = cq.Workplane("XZ").moveTo(0, 0).lineTo({num(R)}, 0).lineTo({num(R)}, {num(Hc)}).threePointArc(({num(m)}, {num(Hc + m)}), (0, {num(Hc + R)})).close().revolve(360, (0, 0, 0), (0, 1, 0))',
            ]
        geo = _geo(
            tg,
            f"{tg.o('This is', 'The part is')} a round-nosed pin: a cylinder with {tg.dia(D)} standing on the XY plane with its axis along Z, {tg.mm(Hc)} high, capped by a hemispherical dome of the same radius ({tg.mm(R)}). The overall height is {tg.mm(Hc + R)}.",
        )
        seq = _seq(
            ts,
            f"On the XY plane, sketch a circle with {ts.dia(D)} centered at the origin and extrude it {ts.mm(Hc)} in +Z.",
            f"Add a hemisphere of radius {ts.mm(R)} on the top face, centered on the axis at Z = {ts.n(Hc)}, so that the dome is tangent to the cylinder wall and the overall height becomes {ts.mm(Hc + R)}.",
        )
        vol, bbox, dims = PI * R * R * Hc + 2 / 3 * PI * R**3, (D, D, Hc + R), {"Hc": Hc, "R": R}
    return Part("revolved", _script(rng, v, lines), geo, seq, bbox, vol, dims)


def loft(rng) -> Part:
    a, b, H = nice(rng, 20, 100, 5), nice(rng, 20, 100, 5), nice(rng, 15, 80, 5)
    kind = rng.choice(("rect_circle", "rect_circle", "rect_rect", "circle_rect"))
    v = V(rng)
    tg, ts = Txt(rng), Txt(rng)
    hh = v("height", H)
    vol = None
    if kind == "rect_rect":
        a2, b2 = nice(rng, 10, 100, 5), nice(rng, 10, 100, 5)
        if (a2, b2) == (a, b):
            raise Reject
        first, second = f"rect({v('base_x', a)}, {v('base_y', b)})", f"rect({v('top_x', a2)}, {v('top_y', b2)})"
        gb, gt_ = f"a rectangle {tg.mm(a)} along X by {tg.mm(b)} along Y", f"a rectangle {tg.mm(a2)} along X by {tg.mm(b2)} along Y"
        sb, st = f"a rectangle {ts.mm(a)} (X) by {ts.mm(b)} (Y)", f"a rectangle {ts.mm(a2)} (X) by {ts.mm(b2)} (Y)"
        vol = H / 6 * (a * b + a2 * b2 + (a + a2) * (b + b2))
        bbox, dims, name = (max(a, a2), max(b, b2), H), {"a": a, "b": b, "a2": a2, "b2": b2, "H": H}, "truncated rectangular pyramid (hopper shape)"
    else:
        d = nice(rng, 10, 100, 5)
        circ = f"circle({v.e(f'{v("circle_d", d)} / 2', d / 2)})"
        rect = f"rect({v('rect_x', a)}, {v('rect_y', b)})"
        first, second = (rect, circ) if kind == "rect_circle" else (circ, rect)
        gr, gc = f"a rectangle {tg.mm(a)} along X by {tg.mm(b)} along Y", f"a circle with {tg.dia(d)}"
        sr, sc = f"a rectangle {ts.mm(a)} (X) by {ts.mm(b)} (Y)", f"a circle with {ts.dia(d)}"
        (gb, gt_), (sb, st) = ((gr, gc), (sr, sc)) if kind == "rect_circle" else ((gc, gr), (sc, sr))
        bbox, dims, name = (max(a, d), max(b, d), H), {"a": a, "b": b, "d_c": d, "H": H}, "transition piece (loft) between a rectangular end and a circular end"
    lines = ["# Two profiles on parallel planes, joined by a loft", f'result = cq.Workplane("XY").{first}.workplane(offset={hh}).{second}.loft(combine=True)']
    geo = _geo(
        tg,
        f"{tg.o('This is', 'The part is')} a {name}, with its axis along Z. The bottom profile is {gb}, centered on the origin in the XY plane (Z = 0). The top profile is {gt_}, centered on the Z-axis at Z = {tg.n(H)}.",
        f"The solid is the straight loft between the two profiles, {tg.mm(H)} high, with planar bottom and top faces.",
    )
    seq = _seq(
        ts,
        f"On the XY plane, sketch {sb} centered at the origin. Create a parallel work plane offset {ts.mm(H)} in +Z and sketch {st} on it, centered on the Z-axis (rectangle sides parallel to X and Y).",
        "Loft between the two sketches to create the solid.",
    )
    return Part("loft", _script(rng, v, lines), geo, seq, bbox, vol, dims)


def swept_elbow(rng) -> Part:
    od = nice(rng, 8, 40, 2)
    wall = nice(rng, 1, 4, 0.5) if rng.random() < 0.6 else 0
    idv = od - 2 * wall if wall else 0
    R = nice(rng, max(od, 15), 80, 5)
    L1 = nice(rng, 10, 80, 5) if rng.random() < 0.7 else 0
    L2 = nice(rng, 10, 80, 5) if rng.random() < 0.7 else 0
    if wall and idv < 3:
        raise Reject
    m = R * math.sqrt(0.5)
    v = V(rng)
    path = 'cq.Workplane("XZ")' + (f".lineTo(0, {num(L1)})" if L1 else "") + f".threePointArc(({num(R - m)}, {num(L1 + m)}), ({num(R)}, {num(L1 + R)}))" + (f".lineTo({num(R + L2)}, {num(L1 + R)})" if L2 else "")
    prof = f".circle({v.e(f'{v("outer_d", od)} / 2', od / 2)})" + (f".circle({v.e(f'{v("inner_d", idv)} / 2', idv / 2)})" if wall else "")
    lines = ["# Centerline in the XZ plane: up along +Z, 90-degree bend, then along +X", f"path = {path}", "# Circular section swept along the path", f'result = cq.Workplane("XY"){prof}.sweep(path)']
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)
    sec = (lambda t: f"an annular cross-section with {t.dia(od).replace('a ', 'an outer ', 1)} and an inner diameter of {t.mm(idv)} (wall thickness {t.mm(wall)})") if wall else (lambda t: f"a solid circular cross-section with {t.dia(od)}")
    geo = _geo(
        tg,
        f"{tg.o('This is', 'The part is')} a {'pipe' if wall else 'solid round bar'} bent through 90 degrees, with {sec(tg)}. Its centerline lies in the XZ plane and starts at the origin, where the first end face lies on the XY plane.",
        (f"The centerline first runs straight up along +Z for {tg.mm(L1)}, then " if L1 else "The centerline immediately ")
        + f"bends through a 90-degree arc with a centerline radius of {tg.mm(R)} towards +X (arc center at X = {tg.n(R)}, Z = {tg.n(L1)})"
        + (f", and finishes with a straight section {tg.mm(L2)} long along +X." if L2 else "."),
        f"The other end face is perpendicular to the X-axis at X = {tg.n(R + L2)}, centered at Z = {tg.n(L1 + R)}.",
    )
    seq = _seq(
        ts,
        f"On the XY plane, sketch a circle with {ts.dia(od)} centered at the origin" + (f" and a concentric circle of diameter {ts.mm(idv)} to make a ring (wall {ts.mm(wall)})." if wall else "."),
        "On the XZ plane, draw the sweep path starting at the origin: "
        + (f"a line {ts.mm(L1)} long in the +Z direction, then " if L1 else "")
        + f"a tangent 90-degree arc with a radius of {ts.mm(R)} turning towards +X and ending at (X = {ts.n(R)}, Z = {ts.n(L1 + R)})"
        + (f", then a tangent line {ts.mm(L2)} long in the +X direction." if L2 else "."),
        "Sweep the circular profile along the path to create the solid.",
    )
    vol = PI / 4 * (od * od - idv * idv) * (L1 + PI * R / 2 + L2)
    dims = {"d_od": od, "R": R} | ({"wall": wall} if wall else {}) | ({"L1": L1} if L1 else {}) | ({"L2": L2} if L2 else {})
    return Part("swept_elbow", code, geo, seq, (R + L2 + od / 2, od, L1 + R + od / 2), vol, dims)


def gusset_bracket(rng) -> Part:
    A, B, W, t = nice(rng, 40, 100, 5), nice(rng, 40, 100, 5), nice(rng, 30, 80, 5), nice(rng, 4, 10)
    g, tr = nice(rng, 10, min(A, B) - t - 5, 5), nice(rng, 3, 8)
    two = rng.random() < 0.4
    yo = nice(rng, tr / 2 + 4, W / 2 - tr / 2, 1) if two else 0
    v = V(rng)
    a, b, w, tt, gg, trv = v("leg_x", A), v("leg_z", B), v("width", W), v("thickness", t), v("rib_leg", g), v("rib_t", tr)
    tri = f"[({tt}, {tt}), ({v.e(f'{tt} + {gg}', t + g)}, {tt}), ({tt}, {v.e(f'{tt} + {gg}', t + g)})]"
    lines = [
        "# L-shaped body: horizontal base and vertical wall",
        f'base = cq.Workplane("XY").box({a}, {w}, {tt}, centered=(False, True, False))',
        f'wall = cq.Workplane("XY").box({tt}, {w}, {b}, centered=(False, True, False))',
        "# Triangular stiffening rib in the inside corner",
        f'rib = cq.Workplane("XZ").polyline({tri}).close().extrude({v.e(f"{trv} / 2", tr / 2)}, both=True)',
        f"result = base.union(wall).union(rib.translate((0, {num(yo)}, 0))).union(rib.translate((0, {num(-yo)}, 0)))" if two else "result = base.union(wall).union(rib)",
    ]
    code = _script(rng, v, lines)
    tg, ts = Txt(rng), Txt(rng)
    ribs = (lambda t_: f"two such ribs, with their mid-planes at Y = +{t_.n(yo)} and Y = -{t_.n(yo)}") if two else (lambda t_: "one such rib, centered on the mid-plane Y = 0")
    geo = _geo(
        tg,
        f"{tg.o('This is', 'The part is')} an L-shaped bracket reinforced by {'two triangular gussets' if two else 'a triangular gusset'}. The horizontal leg lies on the XY plane and extends {tg.mm(A)} along +X from the outer corner; "
        f"the vertical leg rises to a height of {tg.mm(B)} along +Z. Both legs are {tg.mm(t)} thick and {tg.mm(W)} wide along Y, symmetric about the XZ plane; the outer corner edge lies on the Y-axis.",
        f"Each gusset is a right-triangular rib {tg.mm(tr)} thick (along Y) that fills the inside corner: its two perpendicular sides run {tg.mm(g)} along the top of the horizontal leg and {tg.mm(g)} up the inner face of the vertical leg. There {'are' if two else 'is'} {ribs(tg)}.",
    )
    seq = _seq(
        ts,
        f"Create the horizontal leg as a box from X = 0 to {ts.n(A)}, Z = 0 to {ts.n(t)}, and {ts.mm(W)} wide, centered on Y = 0. Create the vertical leg as a box from X = 0 to {ts.n(t)}, Z = 0 to {ts.n(B)}, with the same width, and merge the two.",
        f"On the XZ plane, sketch a right triangle with vertices (X, Z) = ({ts.n(t)}, {ts.n(t)}), ({ts.n(t + g)}, {ts.n(t)}) and ({ts.n(t)}, {ts.n(t + g)}), i.e. two legs of {ts.mm(g)}; extrude it symmetrically to a thickness of {ts.mm(tr)} and merge it with the bracket. Use {ribs(ts)}.",
    )
    vol = (A * t + (B - t) * t) * W + (2 if two else 1) * g * g / 2 * tr
    return Part("gusset_bracket", code, geo, seq, (A, W, B), vol, {"A": A, "B": B, "W": W, "t": t, "g": g, "tr": tr})


def _notch_area(R, w, depth) -> float:
    return w / 2 * math.sqrt(R * R - w * w / 4) + R * R * math.asin(w / (2 * R)) - (R - depth) * w


def polar_pattern(rng) -> Part:
    OD, T = nice(rng, 40, 120, 5), nice(rng, 4, 20)
    R = OD / 2
    b = nice(rng, 6, OD / 3, 2)
    v = V(rng)
    tg, ts = Txt(rng), Txt(rng)
    od, tt = v("outer_d", OD), v("thickness", T)
    ro = v.e(f"{od} / 2", R)
    vol = None
    if rng.random() < 0.55:
        n, nd = rng.choice((4, 6, 8, 10, 12)), nice(rng, 3, 10)
        nw = nice(rng, 3, 2 * PI * (R - nd) / n * 0.55)
        if R - nd < b / 2 + 4:
            raise Reject
        lines = ["# Disc", f'result = cq.Workplane("XY").circle({ro}).extrude({tt})']
        if rng.random() < 0.5:
            lines += [
                f"# {n} rectangular notches around the rim",
                f"for i in range({n}):",
                f'    notch = cq.Workplane("XY").box({num(2 * nd)}, {num(nw)}, {tt}, centered=(True, True, False)).translate(({ro}, 0, 0))',
                f"    result = result.cut(notch.rotate((0, 0, 0), (0, 0, 1), i * 360 / {n}))",
            ]
        else:
            lines += [f"# {n} rectangular notches around the rim", f'result = result.faces(">Z").workplane().polarArray({ro}, 0, 360, {n}).rect({num(2 * nd)}, {num(nw)}).cutThruAll()']
        lines += ["# Central bore", f'result = result.faces(">Z").workplane().hole({v("bore_d", b)})']
        half, step = math.asin(nw / 2 / R), 2 * PI / n
        ang = [k * PI / 1800 for k in range(3600) if abs((k * PI / 1800 + step / 2) % step - step / 2) >= half] + [i * step + s * half for i in range(n) for s in (-1, 1)]
        bbox = (2 * R * max(math.cos(a) for a in ang), 2 * R * max(math.sin(a) for a in ang), T)
        vol = (PI * R * R - n * _notch_area(R, nw, nd) - PI / 4 * b * b) * T
        geo = _geo(
            tg,
            f"{tg.o('This is', 'The part is')} a notched disc (like a simple spline or indexing wheel) whose axis is the Z-axis: {tg.dia(OD).replace('a ', 'an outer ', 1)}, {tg.mm(T)} thick, with its bottom face on the XY plane and a central bore with {tg.dia(b)}.",
            f"{n} identical rectangular notches are cut into the rim through the full thickness, equally spaced every {num(360 / n)} degrees, the first one centered on the +X axis. "
            f"Each notch is {tg.mm(nw)} wide (tangentially) and {tg.mm(nd)} deep, measured radially from the outer diameter along the notch centerline, so its flat bottom is {tg.mm(R - nd)} from the axis.",
        )
        seq = _seq(
            ts,
            f"On the XY plane, sketch a circle with {ts.dia(OD)} centered at the origin and extrude it {ts.mm(T)} in +Z.",
            f"On the top face, sketch a rectangle {ts.mm(2 * nd)} long in the radial direction and {ts.mm(nw)} wide, centered on the rim at the point where it crosses the +X axis (so it reaches {ts.mm(nd)} into the disc), and cut it through all. Pattern this cut {n} times around the Z-axis at equal {num(360 / n)}-degree intervals.",
            f"Cut a central through-hole with {ts.dia(b)}.",
        )
        dims = {"d_od": OD, "T": T, "d_bore": b, "nw": nw, "nd": nd}
        name = "polar_pattern"
    else:
        n, rt, sw = rng.choice((3, 4, 5, 6, 8)), nice(rng, 4, 10), nice(rng, 4, 10)
        HD = nice(rng, b + 8, OD - 2 * rt - 16, 2)
        if sw > HD - 2 or 2 * PI * (HD / 2) / n < sw + 2:
            raise Reject
        lines = [
            "# Rim, hub and spokes share the same thickness",
            f'rim = cq.Workplane("XY").circle({ro}).circle({v.e(f"{ro} - {v("rim_t", rt)}", R - rt)}).extrude({tt})',
            f'hub = cq.Workplane("XY").circle({v.e(f"{v("hub_d", HD)} / 2", HD / 2)}).extrude({tt})',
            "result = rim.union(hub)",
            f"for i in range({n}):",
            f'    spoke = cq.Workplane("XY").box({num(R - rt / 2)}, {v("spoke_w", sw)}, {tt}, centered=(False, True, False))',
            f"    result = result.union(spoke.rotate((0, 0, 0), (0, 0, 1), i * 360 / {n}))",
            "# Central bore",
            f'result = result.faces(">Z").workplane().hole({v("bore_d", b)})',
        ]
        bbox = (OD, OD, T)
        geo = _geo(
            tg,
            f"{tg.o('This is', 'The part is')} a flat spoked wheel whose axis is the Z-axis, {tg.mm(T)} thick, lying on the XY plane (Z from 0 to {tg.n(T)}).",
            f"It consists of an outer rim with {tg.dia(OD).replace('a ', 'an outer ', 1)} and a radial thickness of {tg.mm(rt)} (inner diameter {tg.mm(OD - 2 * rt)}), a central hub with {tg.dia(HD)} and a bore with {tg.dia(b)}, "
            f"and {n} straight spokes {tg.mm(sw)} wide that join the hub to the rim. The spokes are equally spaced every {num(360 / n)} degrees, the first one along the +X axis.",
        )
        seq = _seq(
            ts,
            f"On the XY plane, sketch two concentric circles with diameters {ts.mm(OD)} and {ts.mm(OD - 2 * rt)} and extrude the ring between them (radial thickness {ts.mm(rt)}) {ts.mm(T)} in +Z to form the rim. Extrude a concentric circle with {ts.dia(HD)} to the same height to form the hub.",
            f"Add a radial spoke: a bar {ts.mm(sw)} wide and {ts.mm(T)} thick running from the axis along +X into the rim; pattern it {n} times around the Z-axis at equal {num(360 / n)}-degree intervals and merge everything into one body.",
            f"Cut a central through-hole with {ts.dia(b)}.",
        )
        dims = {"d_od": OD, "T": T, "d_bore": b, "rt": rt, "sw": sw, "d_hub": HD}
        name = "polar_pattern"
    return Part(name, _script(rng, v, lines), geo, seq, bbox, vol, dims)


def boss_plate(rng) -> Part:
    L, W, T = nice(rng, 30, 120, 5), nice(rng, 30, 100, 5), nice(rng, 4, 20)
    h = nice(rng, 5, 60, 5)
    round_boss = rng.random() < 0.65
    v = V(rng)
    lx, wy, tt, hh = v("base_x", L), v("base_y", W), v("base_t", T), v("boss_h", h)
    lines = ["# Base plate, bottom on the XY plane", rng.choice((f'result = cq.Workplane("XY").box({lx}, {wy}, {tt}, centered=(True, True, False))', f'result = cq.Workplane("XY").rect({lx}, {wy}).extrude({tt})'))]
    tg, ts = Txt(rng), Txt(rng)
    if round_boss:
        d = nice(rng, 10, min(L, W) - 6, 2)
        dh = nice(rng, 4, d - 6, 2) if rng.random() < 0.5 and d >= 12 else 0
        sk = f"circle({v.e(f'{v("boss_d", d)} / 2', d / 2)})"
        gboss, sboss = f"a cylinder with {tg.dia(d)}", f"a circle with {ts.dia(d)}"
        area, dims = PI / 4 * d * d, {"d_boss": d}
    else:
        bl, bw = nice(rng, 10, L - 10, 5), nice(rng, 10, W - 10, 5)
        dh = nice(rng, 4, min(bl, bw) - 6, 2) if rng.random() < 0.5 else 0
        sk = f"rect({v('boss_x', bl)}, {v('boss_y', bw)})"
        gboss, sboss = f"a rectangular block {tg.mm(bl)} along X by {tg.mm(bw)} along Y", f"a rectangle {ts.mm(bl)} (X) by {ts.mm(bw)} (Y)"
        area, dims = bl * bw, {"bl": bl, "bw": bw}
    if rng.random() < 0.6:
        lines += ["# Boss on the top face", f'result = result.faces(">Z").workplane().{sk}.extrude({hh})']
    else:
        lines += ["# Boss built on a plane at the top of the plate", f'boss = cq.Workplane("XY").workplane(offset={tt}).{sk}.extrude({hh})', "result = result.union(boss)"]
    if dh:
        lines += ["# Hole through boss and plate", f'result = result.faces(">Z").workplane().hole({v("hole_d", dh)})']
    geo = _geo(
        tg,
        f"{tg.o('This is a composite part', 'The model is a simple stand')}: a rectangular base plate {tg.mm(L)} long (X), {tg.mm(W)} wide (Y) and {tg.mm(T)} thick (Z), centered on the origin in X and Y with its bottom face on the XY plane.",
        f"At the exact center of its top face stands {gboss}, {tg.mm(h)} high, joined to the plate; the overall height is {tg.mm(T + h)}.",
        f"A vertical through-hole with {tg.dia(dh)} runs along the Z-axis through both the boss and the plate." if dh else "",
        plate=True,
    )
    seq = _seq(
        ts,
        _block_seq(ts, L, W, T, 0),
        f"At the center of the top face, sketch {sboss} and extrude it {ts.mm(h)} upward, merging it with the plate.",
        f"On top of the boss, sketch a concentric circle with {ts.dia(dh)} and cut it through the entire part." if dh else "",
    )
    vol = L * W * T + area * h - PI / 4 * dh * dh * (T + h)
    return Part("boss_plate", _script(rng, v, lines), geo, seq, (L, W, T + h), vol, {"L": L, "W": W, "T": T, "h": h} | dims | ({"d_hole": dh} if dh else {}))


def wedge(rng) -> Part:
    A, B, W = nice(rng, 20, 120, 5), nice(rng, 10, 80, 5), nice(rng, 10, 80, 5)
    h2 = nice(rng, 2, B - 5) if rng.random() < 0.5 else 0
    pts = [(0, 0), (A, 0), (A, h2), (0, B)] if h2 else [(0, 0), (A, 0), (0, B)]
    v = V(rng)
    lines = [f"# {'Trapezoidal' if h2 else 'Right-triangle'} profile in the XZ plane, extruded symmetrically along Y", f'result = cq.Workplane("XZ").polyline({_pts(pts)}).close().extrude({v.e(f"{v("width", W)} / 2", W / 2)}, both=True)']
    tg, ts = Txt(rng), Txt(rng)
    shape = f"the vertical back face at X = 0 is {tg.mm(B)} high" + (f", and the sloping top face descends to a height of {tg.mm(h2)} at the front face X = {tg.n(A)}" if h2 else f", and the sloping face runs from its top edge down to the bottom front edge at X = {tg.n(A)}, where the height is zero")
    geo = _geo(
        tg,
        f"{tg.o('This is', 'The part is')} a {'ramp block with a trapezoidal side profile' if h2 else 'wedge (right-triangular prism)'}. Its flat bottom face lies on the XY plane and is {tg.mm(A)} long along +X, starting at X = 0; {shape}.",
        f"The profile is constant across the width: the part is {tg.mm(W)} wide along Y, symmetric about the XZ plane.",
    )
    seq = _seq(
        ts,
        "On the XZ plane, sketch a closed profile through the points (X, Z) = " + ", ".join(f"({ts.n(x)}, {ts.n(z)})" for x, z in pts) + " in mm.",
        f"Extrude it symmetrically about the sketch plane, {ts.mm(W / 2)} to each side, for a total width of {ts.mm(W)} along Y.",
    )
    return Part("wedge", _script(rng, v, lines), geo, seq, (A, W, B), A * (B + h2) / 2 * W, {"A": A, "B": B, "W": W} | ({"h2": h2} if h2 else {}))


FAMILIES = {
    f.__name__: f
    for f in (
        plate_holes, flange, stepped_shaft, polygon_prism, l_bracket, beam_profile, open_box, tube_flange, slotted_block,
        cbore_plate, revolved, loft, swept_elbow, gusset_bracket, polar_pattern, boss_plate, wedge,
    )
}  # fmt: skip


def make_part(family: str, rng: random.Random) -> Part:
    """Sample one part of the given family (resampling infeasible parameter draws)."""
    fn = FAMILIES[family]
    for _ in range(500):
        try:
            return fn(rng)
        except Reject:
            continue
    raise RuntimeError(f"family '{family}' rejected 500 consecutive parameter draws")


def check_part(part: Part, res: dict) -> str | None:
    """Why a verified candidate must be discarded, or None if it is good."""
    if not res.get("ok"):
        return res.get("error_type") or "invalid"
    if res.get("n_solids") != 1:
        return "multi_solid"
    if res.get("open_frac", 0.0) > 0.005:
        return "not_watertight"
    tol = 0.004 * max(part.bbox)  # the tessellated mesh sits slightly inside curved surfaces
    if any(abs(a - b) > tol for a, b in zip(res["bbox"], part.bbox, strict=True)):
        return "bbox_mismatch"
    if part.volume is not None and abs(res["volume"] - part.volume) > 3e-3 * part.volume:
        return "volume_mismatch"
    return None


# ----------------------------------------------------------------------------- dataset
def _candidate(seed: int, split: str, i: int) -> Part:
    """Candidate i of a split depends only on (seed, split, i): fully reproducible."""
    names = list(FAMILIES)
    return make_part(names[i % len(names)], random.Random(f"{seed}/{split}/{i}"))


def generate_dataset(n_train: int = 20000, n_val: int = 500, n_test: int = 400, seed: int = 0, workers: int | None = None) -> dict:
    """Write ``processed/synth/{train,val,test}.jsonl`` (two rows per part: geo and seq) + stats.json.

    Test and val are generated first, from their own random streams, so they do not change
    when ``n_train`` does; splits are disjoint by script hash.
    """
    from ..cad.sandbox import CadExecutor

    out_dir = paths.ensure(paths.processed_dir("synth"))
    seen: set[str] = set()
    fam_stats = {name: Counter() for name in FAMILIES}
    stats: dict = {"source": "synth", "seed": seed, "splits": {}, "parts": {}}
    with CadExecutor(workers=workers, timeout=30.0) as ex:
        for split, n in (("test", n_test), ("val", n_val), ("train", n_train)):
            accepted: list[tuple[Part, dict]] = []
            i = 0
            while len(accepted) < n:
                batch = max(64, int((n - len(accepted)) * 1.1) + 16)
                cands = []
                for j in range(i, i + batch):
                    part = _candidate(seed, split, j)
                    key = hashlib.sha1(part.code.encode()).hexdigest()
                    if key in seen:
                        fam_stats[part.family]["duplicate"] += 1
                        continue
                    seen.add(key)
                    cands.append(part)
                i += batch
                results = ex.run([{"code": p.code, "self_check": 48} for p in cands], progress=f"synth {split}")
                for part, res in zip(cands, results, strict=True):
                    why = check_part(part, res)
                    fam_stats[part.family][why or "ok"] += 1
                    if why is None and len(accepted) < n:
                        accepted.append((part, res))
            with (out_dir / f"{split}.jsonl").open("w") as fh:
                for k, (part, res) in enumerate(accepted):
                    uid = f"{SPLIT_LETTER[split]}{k:07d}"
                    for style in ("geo", "seq"):
                        row = {
                            "id": f"synth/{split}/{uid}-{style}",
                            "uid": uid,
                            "family": part.family,
                            "style": style,
                            "text": getattr(part, style),
                            "code": part.code,
                            "volume": res["volume"],
                            "bbox": res["bbox"],
                            "n_solids": res["n_solids"],
                            "n_tris": res["n_tris"],
                        }
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["splits"][split] = 2 * len(accepted)
            stats["parts"][split] = len(accepted)
    stats["families"] = {}
    for name, c in fam_stats.items():
        tried = sum(v for k, v in c.items() if k != "duplicate")
        stats["families"][name] = {"verified": tried, "ok": c["ok"], "yield": round(c["ok"] / tried, 4) if tried else None, "rejected": {k: v for k, v in sorted(c.items()) if k != "ok"}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats
