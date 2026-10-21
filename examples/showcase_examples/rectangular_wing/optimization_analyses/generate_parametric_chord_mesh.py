import meshio
import numpy as np
from scipy.interpolate import interp1d

def generate_chordwise_mesh(base_msh_path, output_msh_path, n_chord):
    """
    Generates a structured quad mesh with arbitrary even chordwise panel count `n_chord`
    (e.g. 20, 40, 60, 80) preserving the exact spanwise stations, airfoil OML clustering,
    and 4-layer semicircular tip cap topology of the reference OpenVSP Gmsh mesh.
    """
    assert n_chord % 2 == 0, "n_chord must be an even integer"
    n_half = n_chord // 2
    
    m = meshio.read(base_msh_path)
    pts = m.points
    quads = m.cells[0].data
    
    # Extract reference 40-node airfoil ring at y=0 (from strip 0)
    strip0 = quads[0:40]
    ring0_nodes = []
    for k in range(40):
        q_prev = strip0[(k - 1) % 40]
        q_curr = strip0[k]
        shared = np.intersect1d(q_prev, q_curr)
        n_0 = shared[0] if np.isclose(pts[shared[0], 1], 0.0, atol=1e-4) else shared[1]
        ring0_nodes.append(n_0)
    
    ref_xz = pts[ring0_nodes][:, [0, 2]] # shape (40, 2): k=0 is TE(1,0), k=1..19 lower, k=20 LE(0,0), k=21..39 upper
    ref_xz_closed = np.vstack([ref_xz, ref_xz[0:1]])
    
    # Parameterize lower surface (s in [0, 20]) and upper surface (s in [20, 40])
    s_ref = np.arange(41, dtype=float)
    fx = interp1d(s_ref, ref_xz_closed[:, 0], kind='cubic')
    fz = interp1d(s_ref, ref_xz_closed[:, 1], kind='cubic')
    
    s_new = np.linspace(0.0, 40.0, n_chord + 1)[:-1] # length n_chord
    new_x = fx(s_new)
    new_z = fz(s_new)
    new_x[0], new_z[0] = 1.0, 0.0
    new_x[n_half], new_z[n_half] = 0.0, 0.0
    for k in range(1, n_half):
        x_avg = 0.5 * (new_x[k] + new_x[n_chord - k])
        z_mag = 0.5 * (abs(new_z[k]) + abs(new_z[n_chord - k]))
        new_x[k] = x_avg
        new_x[n_chord - k] = x_avg
        new_z[k] = -z_mag
        new_z[n_chord - k] = +z_mag

    n_quads_total = len(quads)
    n_strips_half = (n_quads_total - 152) // (2 * 40)
    
    y_right = [0.0]
    for s in range(n_strips_half):
        strip = quads[s*40 : (s+1)*40]
        y_vals = np.unique(np.round(pts[strip.flatten(), 1], 7))
        y_right.append(float(np.max(y_vals)))
        
    y_left = [0.0]
    offset_left_quads = n_strips_half * 40 + 76
    for s in range(n_strips_half):
        strip = quads[offset_left_quads + s*40 : offset_left_quads + (s+1)*40]
        y_vals = np.unique(np.round(pts[strip.flatten(), 1], 7))
        y_left.append(float(np.min(y_vals)))

    new_pts = []
    new_quads = []
    
    # 1. Right wing stations s = 0 .. n_strips_half (y >= 0)
    right_station_node_rings = []
    for s, y_val in enumerate(y_right):
        ring_indices = []
        for k in range(n_chord):
            idx = len(new_pts)
            new_pts.append([new_x[k], y_val, new_z[k]])
            ring_indices.append(idx)
        right_station_node_rings.append(ring_indices)
        
    for s in range(n_strips_half):
        ring_a = right_station_node_rings[s]     # inboard y_a
        ring_b = right_station_node_rings[s + 1] # outboard y_b
        for k in range(n_chord):
            kp1 = (k + 1) % n_chord
            new_quads.append([ring_b[k], ring_a[k], ring_a[kp1], ring_b[kp1]])
            
    # Helper to build 4-layer semicircular tip cap quads
    def build_tip_cap(junc_ring, y_tip, is_right=True):
        low45 = []
        up45 = []
        mid90 = []
        c45 = np.cos(np.pi / 4.0)
        sign_y = +1.0 if is_right else -1.0
        
        for k in range(1, n_half):
            r_k = abs(new_z[k])
            xk = new_x[k]
            idx = len(new_pts)
            new_pts.append([xk, y_tip + sign_y * r_k * c45, -r_k * c45])
            low45.append(idx)
        for k in range(1, n_half):
            # up45 goes from near-LE (n_half + 1) to near-TE (n_chord - 1)
            k_up = n_half + k
            r_k = abs(new_z[k_up])
            xk = new_x[k_up]
            idx = len(new_pts)
            new_pts.append([xk, y_tip + sign_y * r_k * c45, +r_k * c45])
            up45.append(idx)
        for k in range(1, n_half):
            r_k = abs(new_z[k])
            xk = new_x[k]
            idx = len(new_pts)
            new_pts.append([xk, y_tip + sign_y * r_k, 0.0])
            mid90.append(idx)
            
        te_node = junc_ring[0]
        le_node = junc_ring[n_half]
        cap_quads = []
        m_idx = (n_half - 1) // 2
        
        # Layer 1: lower junction (junc_ring[0..n_half]) to low45 (length n_half - 1)
        for i in range(n_half - 1):
            if i < m_idx:
                q = [junc_ring[i], junc_ring[i+1], low45[i+1], low45[i]]
            elif i == m_idx:
                q = [junc_ring[i], junc_ring[i+1], junc_ring[i+2], low45[i]]
            else:
                q = [low45[i-1], junc_ring[i+1], junc_ring[i+2], low45[i]]
            cap_quads.append(q if is_right else q[::-1])
            
        # Layer 2: junc_up (junc_ring[n_half .. n_chord], where n_chord wraps to 0) and up45 (length n_half - 1)
        junc_up = [junc_ring[n_half + i] for i in range(n_half)] + [te_node]
        for i in range(n_half - 1):
            if i < m_idx:
                q = [junc_up[i], junc_up[i+1], up45[i+1], up45[i]]
            elif i == m_idx:
                q = [junc_up[i], junc_up[i+1], junc_up[i+2], up45[i]]
            else:
                q = [up45[i-1], junc_up[i+1], junc_up[i+2], up45[i]]
            cap_quads.append(q if is_right else q[::-1])

        # Layer 3: low45 to mid90 (sharing te_node at start and le_node at end)
        low45_ext = [te_node] + low45 + [le_node]
        for i in range(n_half - 1):
            if i < m_idx:
                q = [low45_ext[i], low45_ext[i+1], mid90[i+1], mid90[i]]
            elif i == m_idx:
                q = [low45_ext[i], low45_ext[i+1], low45_ext[i+2], mid90[i]]
            else:
                q = [mid90[i-1], low45_ext[i+1], low45_ext[i+2], mid90[i]]
            cap_quads.append(q if is_right else q[::-1])

        # Layer 4: up45 to mid90 (from le_node at i=0 back to te_node at i=n_half)
        up45_ext = [le_node] + up45 + [te_node]
        mid90_rev = mid90[::-1]
        for i in range(n_half - 1):
            if i < m_idx:
                q = [up45_ext[i], up45_ext[i+1], mid90_rev[i+1], mid90_rev[i]]
            elif i == m_idx:
                q = [up45_ext[i], up45_ext[i+1], up45_ext[i+2], mid90_rev[i]]
            else:
                q = [mid90_rev[i-1], up45_ext[i+1], up45_ext[i+2], mid90_rev[i]]
            cap_quads.append(q if is_right else q[::-1])
            
        return cap_quads

    new_quads.extend(build_tip_cap(right_station_node_rings[-1], y_right[-1], is_right=True))
    
    # 3. Left wing stations s = 1 .. n_strips_half (y < 0; station 0 at y=0 is shared with right_station_node_rings[0]!)
    left_station_node_rings = [right_station_node_rings[0]]
    for s in range(1, n_strips_half + 1):
        y_val = y_left[s]
        ring_indices = []
        for k in range(n_chord):
            idx = len(new_pts)
            new_pts.append([new_x[k], y_val, new_z[k]])
            ring_indices.append(idx)
        left_station_node_rings.append(ring_indices)
        
    for s in range(n_strips_half):
        ring_a = left_station_node_rings[s + 1]
        ring_b = left_station_node_rings[s]
        for k in range(n_chord):
            kp1 = (k + 1) % n_chord
            new_quads.append([ring_b[k], ring_a[k], ring_a[kp1], ring_b[kp1]])
            
    new_quads.extend(build_tip_cap(left_station_node_rings[-1], y_left[-1], is_right=False))
    
    new_pts_arr = np.array(new_pts, dtype=float)
    new_quads_arr = np.array(new_quads, dtype=np.int32)
    
    out_mesh = meshio.Mesh(points=new_pts_arr, cells=[("quad", new_quads_arr)])
    meshio.write(output_msh_path, out_mesh, file_format="gmsh22", binary=True)
    print(f"Generated {output_msh_path} (n_chord={n_chord}): {len(new_pts_arr)} pts, {len(new_quads_arr)} quads")
    return out_mesh


def generate_arbitrary_mesh(base_msh_path, output_msh_path, n_span, n_chord):
    """
    Generates a structured quad mesh with arbitrary spanwise section count `n_span`
    (e.g. 15, 35, 40, 50, 60) and chordwise panel count `n_chord` (e.g. 40, 50, 60)
    preserving OpenVSP spanwise clustering, airfoil OML clustering, and tip cap topology.
    """
    assert n_chord % 2 == 0, "n_chord must be an even integer"
    assert n_span >= 5, "n_span must be at least 5"
    n_half = n_chord // 2
    
    m = meshio.read(base_msh_path)
    pts = m.points
    quads = m.cells[0].data
    
    # Extract reference 40-node airfoil ring at y=0 (from strip 0)
    strip0 = quads[0:40]
    ring0_nodes = []
    for k in range(40):
        q_prev = strip0[(k - 1) % 40]
        q_curr = strip0[k]
        shared = np.intersect1d(q_prev, q_curr)
        n_0 = shared[0] if np.isclose(pts[shared[0], 1], 0.0, atol=1e-4) else shared[1]
        ring0_nodes.append(n_0)
    
    ref_xz = pts[ring0_nodes][:, [0, 2]]
    ref_xz_closed = np.vstack([ref_xz, ref_xz[0:1]])
    
    s_ref = np.arange(41, dtype=float)
    fx = interp1d(s_ref, ref_xz_closed[:, 0], kind='cubic')
    fz = interp1d(s_ref, ref_xz_closed[:, 1], kind='cubic')
    
    s_new = np.linspace(0.0, 40.0, n_chord + 1)[:-1]
    new_x = fx(s_new)
    new_z = fz(s_new)
    new_x[0], new_z[0] = 1.0, 0.0
    new_x[n_half], new_z[n_half] = 0.0, 0.0
    for k in range(1, n_half):
        x_avg = 0.5 * (new_x[k] + new_x[n_chord - k])
        z_mag = 0.5 * (abs(new_z[k]) + abs(new_z[n_chord - k]))
        new_x[k] = x_avg
        new_x[n_chord - k] = x_avg
        new_z[k] = -z_mag
        new_z[n_chord - k] = +z_mag

    n_quads_total = len(quads)
    n_strips_ref = (n_quads_total - 152) // (2 * 40)
    
    ref_y_right = [0.0]
    for s in range(n_strips_ref):
        strip = quads[s*40 : (s+1)*40]
        y_vals = np.unique(np.round(pts[strip.flatten(), 1], 7))
        ref_y_right.append(float(np.max(y_vals)))
    ref_y_right = np.array(ref_y_right)

    u_ref = np.linspace(0.0, 1.0, len(ref_y_right))
    fy = interp1d(u_ref, ref_y_right, kind='cubic')
    u_new = np.linspace(0.0, 1.0, n_span)
    y_right = fy(u_new)
    y_right[0] = 0.0
    y_right[-1] = 5.0
    y_left = -y_right
    n_strips_half = n_span - 1

    new_pts = []
    new_quads = []
    
    # 1. Right wing stations s = 0 .. n_strips_half (y >= 0)
    right_station_node_rings = []
    for s, y_val in enumerate(y_right):
        ring_indices = []
        for k in range(n_chord):
            idx = len(new_pts)
            new_pts.append([new_x[k], y_val, new_z[k]])
            ring_indices.append(idx)
        right_station_node_rings.append(ring_indices)
        
    for s in range(n_strips_half):
        ring_a = right_station_node_rings[s]     # inboard y_a
        ring_b = right_station_node_rings[s + 1] # outboard y_b
        for k in range(n_chord):
            kp1 = (k + 1) % n_chord
            new_quads.append([ring_b[k], ring_a[k], ring_a[kp1], ring_b[kp1]])
            
    # Helper to build 4-layer semicircular tip cap quads
    def build_tip_cap(junc_ring, y_tip, is_right=True):
        low45 = []
        up45 = []
        mid90 = []
        c45 = np.cos(np.pi / 4.0)
        sign_y = +1.0 if is_right else -1.0
        
        for k in range(1, n_half):
            r_k = abs(new_z[k])
            xk = new_x[k]
            idx = len(new_pts)
            new_pts.append([xk, y_tip + sign_y * r_k * c45, -r_k * c45])
            low45.append(idx)
        for k in range(1, n_half):
            k_up = n_half + k
            r_k = abs(new_z[k_up])
            xk = new_x[k_up]
            idx = len(new_pts)
            new_pts.append([xk, y_tip + sign_y * r_k * c45, +r_k * c45])
            up45.append(idx)
        for k in range(1, n_half):
            r_k = abs(new_z[k])
            xk = new_x[k]
            idx = len(new_pts)
            new_pts.append([xk, y_tip + sign_y * r_k, 0.0])
            mid90.append(idx)
            
        te_node = junc_ring[0]
        le_node = junc_ring[n_half]
        cap_quads = []
        m_idx = (n_half - 1) // 2
        
        # Layer 1
        for i in range(n_half - 1):
            if i < m_idx:
                q = [junc_ring[i], junc_ring[i+1], low45[i+1], low45[i]]
            elif i == m_idx:
                q = [junc_ring[i], junc_ring[i+1], junc_ring[i+2], low45[i]]
            else:
                q = [low45[i-1], junc_ring[i+1], junc_ring[i+2], low45[i]]
            cap_quads.append(q if is_right else q[::-1])
            
        # Layer 2
        junc_up = [junc_ring[n_half + i] for i in range(n_half)] + [te_node]
        for i in range(n_half - 1):
            if i < m_idx:
                q = [junc_up[i], junc_up[i+1], up45[i+1], up45[i]]
            elif i == m_idx:
                q = [junc_up[i], junc_up[i+1], junc_up[i+2], up45[i]]
            else:
                q = [up45[i-1], junc_up[i+1], junc_up[i+2], up45[i]]
            cap_quads.append(q if is_right else q[::-1])

        # Layer 3
        low45_ext = [te_node] + low45 + [le_node]
        for i in range(n_half - 1):
            if i < m_idx:
                q = [low45_ext[i], low45_ext[i+1], mid90[i+1], mid90[i]]
            elif i == m_idx:
                q = [low45_ext[i], low45_ext[i+1], low45_ext[i+2], mid90[i]]
            else:
                q = [mid90[i-1], low45_ext[i+1], low45_ext[i+2], mid90[i]]
            cap_quads.append(q if is_right else q[::-1])

        # Layer 4
        up45_ext = [le_node] + up45 + [te_node]
        mid90_rev = mid90[::-1]
        for i in range(n_half - 1):
            if i < m_idx:
                q = [up45_ext[i], up45_ext[i+1], mid90_rev[i+1], mid90_rev[i]]
            elif i == m_idx:
                q = [up45_ext[i], up45_ext[i+1], up45_ext[i+2], mid90_rev[i]]
            else:
                q = [mid90_rev[i-1], up45_ext[i+1], up45_ext[i+2], mid90_rev[i]]
            cap_quads.append(q if is_right else q[::-1])
            
        return cap_quads

    new_quads.extend(build_tip_cap(right_station_node_rings[-1], y_right[-1], is_right=True))
    
    # 3. Left wing stations
    left_station_node_rings = [right_station_node_rings[0]]
    for s in range(1, n_strips_half + 1):
        y_val = y_left[s]
        ring_indices = []
        for k in range(n_chord):
            idx = len(new_pts)
            new_pts.append([new_x[k], y_val, new_z[k]])
            ring_indices.append(idx)
        left_station_node_rings.append(ring_indices)
        
    for s in range(n_strips_half):
        ring_a = left_station_node_rings[s + 1]
        ring_b = left_station_node_rings[s]
        for k in range(n_chord):
            kp1 = (k + 1) % n_chord
            new_quads.append([ring_b[k], ring_a[k], ring_a[kp1], ring_b[kp1]])
            
    new_quads.extend(build_tip_cap(left_station_node_rings[-1], y_left[-1], is_right=False))
    
    new_pts_arr = np.array(new_pts, dtype=float)
    new_quads_arr = np.array(new_quads, dtype=np.int32)
    
    out_mesh = meshio.Mesh(points=new_pts_arr, cells=[("quad", new_quads_arr)])
    meshio.write(output_msh_path, out_mesh, file_format="gmsh22", binary=True)
    print(f"Generated {output_msh_path} (n_span={n_span}, n_chord={n_chord}): {len(new_pts_arr)} pts, {len(new_quads_arr)} quads")
    return out_mesh


if __name__ == "__main__":
    import VortexAD
    geom_dir = "examples/example_geometries/rectangular_wing"
    if not os.path.exists(geom_dir):
        geom_dir = "examples/example_geometries"
    base = os.path.join(geom_dir, "rectangular_wing_naca0012_35sect.msh")
    for nc in [20, 40, 60]:
        out_p = os.path.join(geom_dir, f"rectangular_wing_naca0012_35sect_{nc}chord.msh")
        m_gen = generate_chordwise_mesh(base, out_p, nc)
        adj = VortexAD.find_cell_adjacency(points=m_gen.points, cells=m_gen.cells_dict)
        te = VortexAD.TE_detection(points=adj[0], cells=adj[1], edges2cells=adj[3], points2cells=adj[4], threshold_theta=125.)
        print(f"  VortexAD TE check for n_chord={nc}: TE nodes={len(te[3])}, upper TE cells={len(te[0])}, lower TE cells={len(te[1])}")
