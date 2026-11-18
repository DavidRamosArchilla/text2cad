---
tags: [concept]
aliases: [ICP, Kabsch, registration]
---
# Rigid alignment (ICP, Kabsch)

Two correct models of the same part can sit in different places or be built along a
different axis. Before comparing them in millimetres (CadQueryEval checks) we find the
**rotation + translation** that puts one on top of the other.

- **Kabsch algorithm**: given matched point pairs, the best rotation comes from a singular
  value decomposition (SVD) of their covariance matrix — exact and fast.
- **ICP (Iterative Closest Point)**: we do not know which point matches which, so
  alternate: (1) match every point to its nearest neighbour in the other cloud, (2) solve
  Kabsch for those pairs, (3) repeat until it stops moving.
- ICP only finds the *nearest* good alignment, so it needs a reasonable start. CAD parts
  usually differ by axis swaps, so [[cad.metrics]] tries the **24 rotations that map axes
  onto axes** plus 4 alignments of the **principal axes** (PCA), runs a short ICP from
  each, keeps the best, and refines it. Deterministic, unlike the random RANSAC start the
  official scorer uses.

The Text2CAD-Bench metrics deliberately do *not* align (orientation is part of the
answer) — see [[Metrics]].

## Learn more
- Wikipedia: Iterative closest point <https://en.wikipedia.org/wiki/Iterative_closest_point> ·
  Kabsch algorithm <https://en.wikipedia.org/wiki/Kabsch_algorithm> · Principal component
  analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>
