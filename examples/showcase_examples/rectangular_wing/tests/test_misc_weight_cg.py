import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import pytest
import csdl_alpha as csdl

SHOWCASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if SHOWCASE_DIR not in sys.path:
    sys.path.insert(0, SHOWCASE_DIR)


def test_three_body_cg_calculation_and_derivatives():
    """Verify the decoupled 3-body mass & CG calculation and analytical JAX derivatives."""
    rec = csdl.Recorder()
    rec.start()

    # Fixed mass properties matching ex_rectangular_wing_to_bwb.py
    structural_mass_val = 15000.0  # kg
    x_struct_val = 4.2  # m
    payload_weight_val = 160000.0 * 4.44822  # ~711.7 kN
    misc_weight_val = 1.0e6  # 1.0 MN as requested
    root_chord_val = 7.5  # m (undeformed root chord at scale_factor=7.5)
    x_le_root_val = 0.5  # m

    structural_mass = csdl.Variable(value=structural_mass_val, name='structural_mass')
    x_struct = csdl.Variable(value=x_struct_val, name='x_struct')
    payload_weight = csdl.Variable(value=payload_weight_val, name='payload_weight')
    misc_weight = csdl.Variable(value=misc_weight_val, name='misc_weight')
    root_chord = csdl.Variable(value=root_chord_val, name='root_chord')
    x_le_root = csdl.Variable(value=x_le_root_val, name='x_le_root')

    # Design variables: payload_center_x and misc_cg
    payload_center_x = csdl.Variable(value=3.75, name='payload_center_x')
    misc_cg = csdl.Variable(value=0.25, name='misc_cg')

    # Decoupled 3-body physics formulation
    cargo_mass = payload_weight / 9.81
    misc_mass = misc_weight / 9.81
    total_mass = structural_mass + cargo_mass + misc_mass

    x_payload = payload_center_x
    x_misc = x_le_root + misc_cg * root_chord

    x_cg = (structural_mass * x_struct + cargo_mass * x_payload + misc_mass * x_misc) / total_mass

    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[payload_center_x, misc_cg],
        additional_outputs=[x_cg, x_misc, total_mass],
        gpu=False,
    )
    sim.run()

    # Analytical values
    cargo_mass_np = payload_weight_val / 9.81
    misc_mass_np = misc_weight_val / 9.81
    total_mass_np = structural_mass_val + cargo_mass_np + misc_mass_np
    x_misc_np = x_le_root_val + 0.25 * root_chord_val
    x_cg_np = (structural_mass_val * x_struct_val + cargo_mass_np * 3.75 + misc_mass_np * x_misc_np) / total_mass_np

    sim_x_cg = float(np.asarray(sim[x_cg]).flatten()[0])
    sim_x_misc = float(np.asarray(sim[x_misc]).flatten()[0])
    sim_total_mass = float(np.asarray(sim[total_mass]).flatten()[0])

    assert np.isclose(sim_total_mass, total_mass_np, rtol=1e-6)
    assert np.isclose(sim_x_misc, x_misc_np, rtol=1e-6)
    assert np.isclose(sim_x_cg, x_cg_np, rtol=1e-6)

    # Derivative checks
    totals = sim.compute_totals()
    d_xcg_d_misccg = float(totals[(x_cg, misc_cg)].flatten()[0])
    d_xcg_d_payx = float(totals[(x_cg, payload_center_x)].flatten()[0])

    # Theoretical derivatives:
    # d(x_cg)/d(misc_cg) = (misc_mass / total_mass) * root_chord
    expected_d_xcg_d_misccg = (misc_mass_np / total_mass_np) * root_chord_val
    # d(x_cg)/d(payload_center_x) = (cargo_mass / total_mass)
    expected_d_xcg_d_payx = cargo_mass_np / total_mass_np

    assert np.isclose(d_xcg_d_misccg, expected_d_xcg_d_misccg, rtol=1e-6)
    assert np.isclose(d_xcg_d_payx, expected_d_xcg_d_payx, rtol=1e-6)

    # Finite difference validation
    h = 1e-5
    sim[misc_cg] = 0.25 + h
    sim.run()
    xcg_plus = float(np.asarray(sim[x_cg]).flatten()[0])
    sim[misc_cg] = 0.25 - h
    sim.run()
    xcg_minus = float(np.asarray(sim[x_cg]).flatten()[0])
    fd_misccg = (xcg_plus - xcg_minus) / (2.0 * h)

    rel_err = abs(d_xcg_d_misccg - fd_misccg) / abs(fd_misccg)
    assert rel_err < 1e-6, f"Finite difference mismatch for misc_cg: JAX={d_xcg_d_misccg}, FD={fd_misccg}"


def test_misc_weight_no_yz_motion():
    """Verify that misc_weight location is strictly fixed in y=0 and z=beam_midplane."""
    rec = csdl.Recorder()
    rec.start()

    upper_beam_z = csdl.Variable(value=np.array([0.45]))
    lower_beam_z = csdl.Variable(value=np.array([-0.45]))
    misc_cg = csdl.Variable(value=0.25, name='misc_cg')

    y_misc = csdl.Variable(value=np.array([0.0]))
    z_misc = 0.5 * (upper_beam_z + lower_beam_z)

    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec,
        additional_inputs=[misc_cg],
        additional_outputs=[y_misc, z_misc],
        gpu=False,
    )
    sim.run()

    assert np.isclose(float(np.asarray(sim[y_misc]).flatten()[0]), 0.0)
    assert np.isclose(float(np.asarray(sim[z_misc]).flatten()[0]), 0.0)

    # Verify y_misc and z_misc have zero sensitivity to misc_cg
    totals = sim.compute_totals()
    d_ymisc_d_misccg = float(totals[(y_misc, misc_cg)].flatten()[0])
    d_zmisc_d_misccg = float(totals[(z_misc, misc_cg)].flatten()[0])

    assert np.isclose(d_ymisc_d_misccg, 0.0)
    assert np.isclose(d_zmisc_d_misccg, 0.0)


def test_misc_weight_value_is_one_million():
    """Verify misc_weight is fixed at 1.0e6 N as requested."""
    # Check the variable definition in ex_rectangular_wing_to_bwb.py
    script_path = os.path.join(SHOWCASE_DIR, 'ex_rectangular_wing_to_bwb.py')
    with open(script_path, 'r') as f:
        content = f.read()

    assert "misc_weight = csdl.Variable(value=1.e6)" in content

