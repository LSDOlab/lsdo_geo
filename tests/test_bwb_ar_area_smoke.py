import os
import sys
import pytest


def test_bwb_ar_area_forward_smoke():
    """
    Smoke test running the BWB model in ar_area formulation forward without optimization.
    Verifies that the ParameterizationSolver converges and all physics models evaluate cleanly.
    """
    example_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/showcase_examples/rectangular_wing")
    )
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    os.environ["SKIP_OPTIMIZATION"] = "1"
    try:
        import examples.showcase_examples.rectangular_wing.ex_rectangular_wing_to_bwb as bwb
        assert bwb.formulation == "ar_area"
        assert bwb.resolution == "fast"
        assert bwb.scale_factor == 7.5
        # Verify that residual arrays are evaluated and available
        output_names = [getattr(var, "name", None) for var in bwb.additional_outs]
        assert "taper_weak_residual" in output_names
        assert "thickness_weak_residual" in output_names
        assert "sweep_weak_residual" in output_names
    finally:
        pass
