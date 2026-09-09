import pytest
import numpy as np
import csdl_alpha as csdl

def test_beam_mesh_panel_mapping():
    """Test aerostructural force and moment mapping from panel centers to beam nodes."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_beam_nodes = 21
    num_panels = 100

    # Build mapping matrix where each panel is mapped to a beam node
    W = np.zeros((num_beam_nodes, num_panels))
    for i in range(num_panels):
        n = i % num_beam_nodes
        W[n, i] = 1.0

    np.random.seed(42)
    W_var = csdl.Variable(value=W)
    beam_mesh = csdl.Variable(value=np.random.rand(num_beam_nodes, 3))
    dynamic_panel_centers = csdl.Variable(value=np.random.rand(num_panels, 3))
    F = csdl.Variable(value=np.random.rand(num_panels, 3))

    B_expand = csdl.expand(beam_mesh, (num_beam_nodes, num_panels, 3), 'nj->nij')
    C_expand = csdl.expand(dynamic_panel_centers, (num_beam_nodes, num_panels, 3), 'ij->nij')
    F_expand = csdl.expand(F, (num_beam_nodes, num_panels, 3), 'ij->nij')
    W_expand = csdl.expand(W_var, (num_beam_nodes, num_panels, 3), 'ni->nij')

    r = C_expand - B_expand
    r_cross_F = csdl.cross(r, F_expand, axis=2)
    M_node = csdl.sum(W_expand * r_cross_F, axes=(1,))
    F_node = csdl.matmat(W_var, F)

    # Check shapes
    assert F_node.shape == (num_beam_nodes, 3)
    assert M_node.shape == (num_beam_nodes, 3)

    # Check conservation of total force (sum of node forces equals sum of panel forces)
    total_node_force = np.sum(F_node.value, axis=0)
    total_panel_force = np.sum(F.value, axis=0)
    np.testing.assert_allclose(total_node_force, total_panel_force, atol=1e-10)

    # Ensure no NaNs
    assert not np.isnan(F_node.value).any()
    assert not np.isnan(M_node.value).any()
