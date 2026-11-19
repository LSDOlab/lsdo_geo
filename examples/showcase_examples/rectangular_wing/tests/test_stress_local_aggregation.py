import pytest
import numpy as np
import lsdo_function_spaces as lfs
import csdl_alpha as csdl


@pytest.mark.parametrize("num_thickness_stations", [5, 8])
def test_stress_local_aggregation_captures_all_elements(num_thickness_stations):
    """Verify that the local stress aggregation captures every beam element center at least once
    and that the number of stations equals num_thickness_stations (num_spanwise_dvs)."""
    # 1. Setup thickness B-spline space and Greville abscissae (thickness_peaks)
    thickness_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=2,
        coefficients_shape=(num_thickness_stations,),
    )
    knots_thick = thickness_space.knots[0]
    thickness_peaks = np.array([(knots_thick[i+1] + knots_thick[i+2]) / 2.0 for i in range(num_thickness_stations)])

    # 2. 14 finite element centers across [0, 1]
    num_beam_elements = 14
    y_norm_elem = ((np.arange(num_beam_elements) + 0.5) / num_beam_elements).reshape((-1, 1))

    # 3. Partition bounds
    midpoints = 0.5 * (thickness_peaks[:-1] + thickness_peaks[1:])
    bounds = np.concatenate([[0.0], midpoints, [1.0]])

    # 4. Dense evaluation grid
    pts_list = [y_norm_elem.flatten(), thickness_peaks]
    for i in range(num_thickness_stations - 1):
        pts_list.append(np.linspace(thickness_peaks[i], thickness_peaks[i+1], 11)[1:-1])
    eval_points_arr = np.sort(np.unique(np.concatenate(pts_list))).reshape((-1, 1))

    # Check that each element center is an exact sample point
    for y_e in y_norm_elem.flatten():
        assert np.any(np.isclose(eval_points_arr.flatten(), y_e, atol=1e-12)), f"Element center {y_e} not in eval_points"

    # 5. Build CSDL graph
    recorder = csdl.Recorder()
    recorder.start()

    stress_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(15,),
    )
    # Mock coefficients
    stress_coeffs = csdl.Variable(value=np.ones((15, 1)) * 150.0e6)
    stress_func = lfs.Function(space=stress_space, coefficients=stress_coeffs)

    all_eval_stresses = stress_func.evaluate(eval_points_arr)

    aggregated_stress_list = []
    covered_elements = set()
    for i in range(num_thickness_stations):
        grp = np.where((eval_points_arr.flatten() >= bounds[i]) & (eval_points_arr.flatten() <= bounds[i+1]))[0]
        assert len(grp) > 0, f"Station {i} has empty evaluation group!"
        
        # Verify which elements fall in this station's aggregation group
        pts_in_grp = eval_points_arr.flatten()[grp]
        for e_idx, e_y in enumerate(y_norm_elem.flatten()):
            if np.any(np.isclose(pts_in_grp, e_y, atol=1e-12)):
                covered_elements.add(e_idx)

        sub_stresses = csdl.concatenate([all_eval_stresses[idx] for idx in grp])
        grp_max = csdl.maximum(sub_stresses, axes=(0,), rho=1.0)
        aggregated_stress_list.append(csdl.reshape(grp_max, (1,)))

    dv_stresses = csdl.concatenate(aggregated_stress_list)

    recorder.stop()
    sim = csdl.experimental.PySimulator(recorder)
    sim.run()

    # Assertions
    # 1. Every single element is captured at least once
    assert len(covered_elements) == num_beam_elements, f"Missing elements: {set(range(num_beam_elements)) - covered_elements}"
    # 2. dv_stresses has shape (num_thickness_stations,)
    assert dv_stresses.shape == (num_thickness_stations,)
    val = sim[dv_stresses].flatten()
    assert len(val) == num_thickness_stations
    assert np.all(np.isfinite(val))
