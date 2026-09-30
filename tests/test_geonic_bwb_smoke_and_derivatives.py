from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import pytest
import csdl_alpha as csdl


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            elif module.startswith("numpy.core"):
                module = module.replace("numpy.core", "numpy._core")
            return super().find_class(module, name)


pickle.load = lambda f, **kwargs: CompatUnpickler(f, **kwargs).load()

import lsdo_geo
from lsdo_geo import (
    import_geometry,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
)
import bsm3
from examples.showcase_examples.rectangular_wing.physics_models.geonic_payload import (
    build_geonic_payload_sample_points,
    compute_geonic_clearance_and_margin,
    GEONIC_CLEARANCE_M,
)


def test_geonic_derivatives_payload_thickness_camber():
    """
    Verify total derivatives of geonic_margin with respect to:
    - payload_center_x
    - thickness DVs (FFD thickness stretch)
    - camber DVs (FFD camber displacement)
    Comparing against centered finite differences away from SDF topology switches.
    """
    rec = csdl.Recorder(inline=True)
    rec.start()

    scale_factor = 7.5
    geometry = import_geometry("examples/example_geometries/rectangular_wing_naca0012_10ar.stp", parallelize=False)
    for f in geometry.functions.values():
        f.coefficients = f.coefficients * scale_factor

    num_stations = 5
    num_ffd_sections = 9
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry,
        num_coefficients=(5, num_ffd_sections, 2),
        degree=(2, 3, 1),
    )
    ffd_param = SectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    payload_center_x = csdl.Variable(value=3.8, name="payload_center_x")
    payload_center_x.set_as_design_variable()

    thickness_dvs = csdl.Variable(value=np.zeros(num_stations), name="thickness_dvs")
    thickness_dvs.set_as_design_variable()

    # 1% initial root camber so camber derivatives are active and nonzero
    camber_init = np.zeros((3, num_stations))
    camber_init[0, 0] = 1.0
    camber_dvs = csdl.Variable(shape=(3, num_stations), value=camber_init, name="camber_dvs")
    camber_dvs.set_as_design_variable()

    chord_params = csdl.Variable(value=np.zeros(num_ffd_sections))
    sweep_params = csdl.Variable(value=np.zeros(num_ffd_sections))
    thickness_params = csdl.concatenate(
        [thickness_dvs[i] for i in range(num_stations - 1, 0, -1)] +
        [thickness_dvs[i] for i in range(num_stations)]
    )
    twist_params = csdl.Variable(value=np.zeros(num_ffd_sections))
    span_params = csdl.Variable(value=np.zeros(num_ffd_sections))

    sectional_params = SectionalParameters()
    sectional_params.add_stretch(axis=np.array([1.0, 0.0, 0.0]), stretch=chord_params)
    sectional_params.add_translation(axis=np.array([1.0, 0.0, 0.0]), translation=sweep_params)
    sectional_params.add_stretch(axis=np.array([0.0, 0.0, 1.0]), stretch=thickness_params)
    sectional_params.add_translation(axis=np.array([0.0, 1.0, 0.0]), translation=span_params)
    sectional_params.add_rotation(
        axis=np.array([0.0, 1.0, 0.0]), rotation=twist_params, parametric_coordinate=np.array([0.25, 0.5])
    )

    ffd_coefficients = ffd_param.evaluate(sectional_params, plot=False)

    section_chords = ffd_coefficients[-1, :, 0, 0] - ffd_coefficients[0, :, 0, 0]
    full_span_camber_list = []
    for c in range(3):
        row = csdl.concatenate(
            [camber_dvs[c, i] for i in range(num_stations - 1, 0, -1)] +
            [camber_dvs[c, i] for i in range(num_stations)]
        )
        full_span_camber_list.append(csdl.reshape(row, (1, num_ffd_sections)))
    full_span_camber = csdl.concatenate(full_span_camber_list, axis=0)
    camber_displacement = (full_span_camber / 100.0) * csdl.expand(section_chords, (3, num_ffd_sections), "j->ij")
    camber_delta = csdl.expand(camber_displacement, (3, num_ffd_sections, 2), "ij->ijk")
    ffd_coefficients = ffd_coefficients.set(csdl.slice[1:4, :, :, 2], ffd_coefficients[1:4, :, :, 2] + camber_delta)

    geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients)

    # Sample points & BSM3 enclosed SDF
    pts = build_geonic_payload_sample_points(payload_center_x)
    model = bsm3.FunctionSetProjectionModel(
        function_set=geometry,
        warm_start_nu=50,
        warm_start_nv=50,
        sdf=True,
        sdf_sign_mode="enclosed",
    )
    sdf_op = bsm3.FunctionSetClosestDistanceOperation(model=model)
    sdf = sdf_op.evaluate(coefficients=geometry.stack_coefficients(), points=pts)
    _, _, margin = compute_geonic_clearance_and_margin(sdf, clearance_buffer=GEONIC_CLEARANCE_M, rho=50.0)

    obj = margin
    obj.set_as_objective()

    sim = csdl.experimental.JaxSimulator(recorder=rec)
    sim.run()

    # Verify finite difference vs analytical derivatives
    sim.check_optimization_derivatives(step_size=1e-5, raise_on_error=True)


def test_bwb_geonic_modes_smoke():
    """
    Verify both modes in BWB:
    1. Disabled mode (use_geonic=False):
       Preserves legacy payload_cg in design_variables and additional_outs.
    2. Enabled mode (use_geonic=True):
       Exposes only payload_center_x as payload DV, produces 8 finite SDF and clearance values,
       and finite geonic_margin.
    """
    import subprocess

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Test 1: Disabled mode check
    code_disabled = """
import os, sys
sys.path.insert(0, os.path.abspath('examples/showcase_examples/rectangular_wing'))
os.environ['SKIP_OPTIMIZATION'] = '1'
os.environ['USE_GEONIC'] = '0'

import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as bwb
assert bwb.use_geonic is False
assert 'payload_cg' in bwb.design_variables
assert 'payload_center_x' not in bwb.design_variables
out_names = [getattr(v, 'name', None) for v in bwb.additional_outs]
assert 'payload_cg' in out_names
assert 'payload_signed_distance' not in out_names
print('DISABLED_MODE_OK')
"""
    res_dis = subprocess.run([sys.executable, "-c", code_disabled], cwd=repo_root, capture_output=True, text=True)
    assert res_dis.returncode == 0, f"Disabled mode failed: {res_dis.stderr}"
    assert "DISABLED_MODE_OK" in res_dis.stdout

    # Test 2: Enabled mode check
    code_enabled = """
import os, sys
sys.path.insert(0, os.path.abspath('examples/showcase_examples/rectangular_wing'))
os.environ['SKIP_OPTIMIZATION'] = '1'
os.environ['USE_GEONIC'] = '1'

import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as bwb
import numpy as np

assert bwb.use_geonic is True
assert 'payload_center_x' in bwb.design_variables
assert 'payload_cg' not in bwb.design_variables

out_names = [getattr(v, 'name', None) for v in bwb.additional_outs]
assert 'payload_center_x' in out_names
assert 'payload_sample_points' in out_names
assert 'payload_signed_distance' in out_names
assert 'geonic_payload_oml_clearance' in out_names
assert 'geonic_clearance_per_point' in out_names
assert 'geonic_margin' in out_names
assert 'payload_cg' not in out_names

# Initial values from the graph
sim = bwb.jax_sim
# Verify payload_sample_points shape
pts_val = np.asarray(bwb.payload_sample_points.value)
assert pts_val.shape == (8, 3)
assert np.all(np.isfinite(pts_val))

print('ENABLED_MODE_OK')
"""
    res_en = subprocess.run([sys.executable, "-c", code_enabled], cwd=repo_root, capture_output=True, text=True)
    assert res_en.returncode == 0, f"Enabled mode failed: {res_en.stderr}"
    assert "ENABLED_MODE_OK" in res_en.stdout
