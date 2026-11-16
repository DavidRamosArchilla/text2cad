"""The verifier's numbers are the foundation of every reported metric and of the RL reward,
so they are checked against shapes whose answers are known analytically."""

import math

import numpy as np
import pytest

from t2c.cad import geometry as G
from t2c.cad import metrics as M

BOX = "import cadquery as cq\nresult = cq.Workplane('XY').box({},{},{})\n"


def mesh_of(executor, code, tmp_path, name="m"):
    path = tmp_path / f"{name}.npz"
    res = executor.run_one({"code": code, "save_mesh": str(path)})
    assert res["ok"], res
    return G.load_mesh(path), res


def occupancy_fraction(mesh, res=128):
    v, f = mesh
    occ, open_frac = G.voxelize(G.normalize_unit_bbox(v), f, res)
    return occ.mean(), open_frac


def test_voxel_volume_box(executor, tmp_path):
    mesh, _ = mesh_of(executor, BOX.format(10, 20, 40), tmp_path)
    frac, open_frac = occupancy_fraction(mesh)
    assert open_frac == 0
    assert frac == pytest.approx((10 / 40) * (20 / 40), rel=0.02)


def test_voxel_volume_sphere_and_cavity(executor, tmp_path):
    sphere, _ = mesh_of(executor, "import cadquery as cq\nresult = cq.Workplane('XY').sphere(10)\n", tmp_path, "s")
    frac, open_frac = occupancy_fraction(sphere, 128)
    assert open_frac == 0
    assert frac == pytest.approx(math.pi / 6, rel=0.02)
    # Fully enclosed void: the even-odd rule must leave it empty (a flood fill would not).
    hollow, _ = mesh_of(
        executor,
        "import cadquery as cq\nresult = cq.Workplane('XY').sphere(10).cut(cq.Workplane('XY').sphere(6))\n",
        tmp_path,
        "h",
    )
    frac, open_frac = occupancy_fraction(hollow, 128)
    assert open_frac == 0
    assert frac == pytest.approx(math.pi / 6 * (1 - 0.6**3), rel=0.03)


def test_voxel_through_hole(executor, tmp_path):
    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(40,40,10).faces('>Z').workplane().hole(20)\n"
    mesh, res = mesh_of(executor, code, tmp_path)
    frac, _ = occupancy_fraction(mesh, 128)
    assert frac == pytest.approx(res["volume"] / 40**3, rel=0.02)


def test_iou_identical_and_scaled(executor, tmp_path):
    a, _ = mesh_of(executor, BOX.format(10, 20, 30), tmp_path, "a")
    b, _ = mesh_of(executor, BOX.format(100, 200, 300), tmp_path, "b")  # same shape, 10x larger
    m = M.protocol_metrics(a, b, res=128, n_points=20000)
    assert m["iou"] == pytest.approx(1.0, abs=1e-6)  # unit-bbox normalisation removes scale
    assert m["cd"] < 0.03  # sampling-noise floor, not zero (independent point sets)
    assert m["dim_err"] == pytest.approx(math.log(10), rel=1e-3)  # ...but dim_err sees it


def test_iou_known_overlap(executor, tmp_path):
    # Unit-normalised: a 1x1x1 cube vs a 1x1x0.5 slab centred in the same box -> IoU = 0.5
    a, _ = mesh_of(executor, BOX.format(10, 10, 10), tmp_path, "a")
    b, _ = mesh_of(executor, BOX.format(10, 10, 5), tmp_path, "b")
    assert M.protocol_metrics(a, b, res=128, n_points=0)["iou"] == pytest.approx(0.5, abs=0.02)


def test_iou_is_orientation_sensitive(executor, tmp_path):
    a, _ = mesh_of(executor, BOX.format(10, 20, 40), tmp_path, "a")
    b, _ = mesh_of(executor, BOX.format(40, 20, 10), tmp_path, "b")
    assert M.protocol_metrics(a, b, res=64, n_points=0)["iou"] < 0.2


def test_open_mesh_is_flagged():
    # A cube with its top face removed.
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    _, open_frac = G.voxelize(v, f, 32)
    assert open_frac > 0.9


def test_rigid_align_recovers_axis_swap_and_offset(executor, tmp_path):
    from scipy.spatial import cKDTree

    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(30,20,10).faces('>Z').workplane().center(8,3).hole(6)\n"
    (v, f), _ = mesh_of(executor, code, tmp_path)
    rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # cyclic axis permutation
    moved = v @ rot.T + np.array([50.0, -20.0, 7.0])
    p, q = G.sample_surface(moved, f, 20000, 0), G.sample_surface(v, f, 20000, 1)
    before, _ = cKDTree(q).query(p)
    r, t = M.rigid_align(p, q, final_corr=1.5)
    after, _ = cKDTree(q).query(p @ r.T + t)
    assert before.mean() > 10
    assert after.mean() < 0.3 and np.percentile(after, 99) < 0.6  # down to point-sampling noise


def test_cqe_checks_pass_and_fail(executor, tmp_path):
    ref, _ = mesh_of(executor, "import cadquery as cq\nresult = cq.Workplane('XY').box(30,20,10).faces('>Z').workplane().hole(6)\n", tmp_path, "r")
    same, _ = mesh_of(executor, "import cadquery as cq\nresult = cq.Workplane('YZ').box(30,20,10).faces('>X').workplane().hole(6)\n", tmp_path, "s")
    wrong, _ = mesh_of(executor, "import cadquery as cq\nresult = cq.Workplane('XY').box(30,20,10).faces('>Z').workplane().hole(12)\n", tmp_path, "w")
    ok = M.cqe_checks(same, ref)
    assert ok["passed"], ok  # same part modelled on another plane: alignment must find it
    bad = M.cqe_checks(wrong, ref)
    assert not bad["passed"] and not bad["volume_passed"]
