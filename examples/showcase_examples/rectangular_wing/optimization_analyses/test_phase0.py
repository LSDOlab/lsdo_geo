import os, sys, time
import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['RUN_OPTIMIZATION'] = 'false'
os.environ['DIAGNOSTIC_ONLY'] = 'true'

cases = [
    ('5DV_15S', 5, 'rectangular_wing_naca0012_10ar', 'none'),
    ('5DV_35S', 5, 'rectangular_wing_naca0012_35sect', 'none'),
    ('5DV_60S', 5, 'rectangular_wing_naca0012_60sect', 'none'),
    ('8DV_15S', 8, 'rectangular_wing_naca0012_10ar', 'none'),
    ('8DV_35S', 8, 'rectangular_wing_naca0012_35sect', 'none'),
    ('6DV_35S', 6, 'rectangular_wing_naca0012_35sect', 'none'),
    ('7DV_35S', 7, 'rectangular_wing_naca0012_35sect', 'none'),
    ('8DV_60S', 8, 'rectangular_wing_naca0012_60sect', 'none'),
]

print("Starting Phase 0 Diagnostic Sweep across configurations...")
for name, n_dv, mesh_name, payload in cases:
    t0 = time.time()
    os.environ['BWB_NUM_STATIONS'] = str(n_dv)
    os.environ['BWB_AERO_MESH_FILE'] = mesh_name
    os.environ['GEONIC_PAYLOAD_MODE'] = payload
    print(f"\n--- Checking {name}: DVs={n_dv}, Mesh={mesh_name}, Payload={payload} ---")
    
    # We run as a subprocess to keep memory clean
    cmd = f"/home/andrewfletcher/anaconda3/envs/py310/bin/python examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py"
    ret = os.system(f"BWB_NUM_STATIONS={n_dv} BWB_AERO_MESH_FILE={mesh_name} GEONIC_PAYLOAD_MODE={payload} RUN_OPTIMIZATION=false DIAGNOSTIC_ONLY=true {cmd} > /tmp/diag_{name}.log 2>&1")
    dt = time.time() - t0
    if ret == 0:
        print(f"SUCCESS: {name} completed in {dt:.1f} s")
    else:
        print(f"FAILED: {name} (exit code {ret}) in {dt:.1f} s. Check /tmp/diag_{name}.log")
