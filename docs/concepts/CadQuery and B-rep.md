---
tags: [concept]
aliases: [CadQuery, B-rep, OpenCascade, OCCT, CSG]
---
# CadQuery and B-rep

**CadQuery** is a Python library for parametric CAD. A script describes how to build a
part, step by step:

```python
import cadquery as cq
result = (
    cq.Workplane("XY")            # start on the XY plane
    .box(40, 30, 10)              # a block
    .faces(">Z").workplane()      # select the top face, sketch on it
    .rect(28, 18, forConstruction=True).vertices()
    .hole(4)                      # four through holes at the rectangle's corners
    .edges("|Z").fillet(2)        # round the vertical edges
)
```

Key ideas: a **Workplane** carries a stack of selected objects; **selectors** like `">Z"`
(the face furthest in +Z) or `"|Z"` (edges parallel to Z) pick geometry; 2D sketches become
solids with `extrude`, `revolve`, `loft`, `sweep`; solids combine with `union`, `cut`,
`intersect` (**CSG**); features like `hole`, `fillet`, `chamfer`, `shell` modify them.

Underneath is **OpenCascade (OCCT)**, a C++ geometry kernel, accessed through the **OCP**
bindings. It stores solids as a **B-rep** (*boundary representation*): exact surfaces
(planes, cylinders, splines) bounded by edges and vertices. To compare two solids we
**tessellate** them into triangle meshes ([[Comparing 3D shapes]]).

## Why it matters here
- It is the output language of every benchmark.
- Typical model mistakes are CadQuery mistakes: invented methods (`Workplane.square`,
  `cq.Circle`), impossible operations (a fillet larger than the edge → `StdFail_NotDone`),
  selecting nothing ("Cannot find a solid on the stack"), forgetting `result`.
- OCCT can crash the process on bad input → each script runs in its own process
  ([[cad.sandbox]]).

## Learn more
- **CadQuery documentation**: <https://cadquery.readthedocs.io/en/latest/> — start with the
  **Quickstart** <https://cadquery.readthedocs.io/en/latest/quickstart.html>, then
  **Concepts** <https://cadquery.readthedocs.io/en/latest/primer.html> and
  **Examples** <https://cadquery.readthedocs.io/en/latest/examples.html>
- Source: <https://github.com/CadQuery/cadquery> · OCP bindings: <https://github.com/CadQuery/OCP>
- OpenCascade: <https://dev.opencascade.org/>
- Wikipedia: Boundary representation <https://en.wikipedia.org/wiki/Boundary_representation> ·
  Constructive solid geometry <https://en.wikipedia.org/wiki/Constructive_solid_geometry>
