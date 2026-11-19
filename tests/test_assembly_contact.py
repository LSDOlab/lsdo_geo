"""Tests for ContactPair and Assembly.evaluate_contact_constraints in lsdo_geo."""

import numpy as np
import csdl_alpha as csdl
import pytest

import lsdo_function_spaces as lfs
from lsdo_function_spaces.core.spaces.b_spline_space import BSplineSpace
import lsdo_geo as lg
from lsdo_geo import Assembly, ContactPair, Geometry


def test_contact_pair_creation():
    """Verify ContactPair creation and registration in Assembly."""
    assembly = Assembly(name="test_assembly")
    pair = assembly.add_contact_pair(
        source="box_inner",
        target="box_outer",
        interaction="interior",
        method="variational",
        formulation="basis_weighted_ks",
        name="inner_in_outer",
    )
    assert pair.name == "inner_in_outer"
    assert pair.interaction == "interior"
    assert pair.method == "variational"
    assert assembly.get_contact_pair("inner_in_outer") is pair


def test_assembly_variational_contact_evaluation():
    """
    Verify evaluate_contact_constraints on an Assembly containing a moving patch
    and an analytical signed distance target.
    """
    # 1. Create a 2D planar patch in R^3 (e.g. z = 0 face)
    space = BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),  # 4 control points
    )
    # Control points on [0, 1] x [0, 1] in XY plane at z=0
    ctrl_pts = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    cp_var = csdl.Variable(value=ctrl_pts)
    fn = lfs.Function(space=space, coefficients=cp_var)
    geom = Geometry(functions={0: fn}, name="patch_geom")

    # 2. Target signed distance: simple box [-0.5, 1.5] x [-0.5, 1.5] x [-1.0, 1.0]
    # Patch [0, 1] x [0, 1] is strictly inside the box. Inside should be negative!
    def box_sdf(pts):
        # Center at (0.5, 0.5, 0)
        center = np.tile(np.array([[0.5, 0.5, 0.0]]), (pts.shape[0], 1))
        diff = pts - center
        r = csdl.sqrt(csdl.sum(diff**2, axes=(1,)) + 1e-12)
        return r - 3.0

    # 3. Create Assembly and register contact
    assembly = Assembly(name="assembly_test")
    assembly.add_component(geom, name="patch_geom")
    assembly.add_contact_pair(
        source="patch_geom",
        target=box_sdf,
        interaction="interior",
        method="variational",
        formulation="basis_weighted_ks",
        rho_local=50.0,
        name="patch_in_sphere",
    )

    # 4. Evaluate contact constraints
    constraints = assembly.evaluate_contact_constraints()
    assert "patch_in_sphere" in constraints
    residuals = constraints["patch_in_sphere"]
    assert "patch_0" in residuals
    R = residuals["patch_0"]

    # Since space has (2, 2) control points, there are exactly 4 basis functions
    assert R.shape == (4,)
    # Patch is inside radius 3 sphere, so signed distance is negative (satisfied <= 0)
    assert np.all(R.value < 0.0)


def test_assembly_rebind_contact():
    """Verify that ContactPair rebinds correctly when Assembly is cloned/rebound."""
    assembly = Assembly(name="original")
    cp = assembly.add_contact_pair(
        source="comp_a",
        target="comp_b",
        name="pair_ab",
    )
    target_map = {"comp_a": "comp_a_cloned", "comp_b": "comp_b_cloned"}
    rebound = assembly.rebind(target_map, name="rebound_assembly")

    rebound_cp = rebound.get_contact_pair("pair_ab")
    assert rebound_cp is not None
    assert rebound_cp.source == "comp_a_cloned"
    assert rebound_cp.target == "comp_b_cloned"
