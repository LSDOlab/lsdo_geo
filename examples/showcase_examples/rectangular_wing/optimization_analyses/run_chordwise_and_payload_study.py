import os, sys, time
import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu'

# Focused Chordwise Refinement Study Queue on 35-section mesh (20-chord and 60-chord vs existing 40-chord):
cases = [
    ('5DV_35S_20C', 5, 'rectangular_wing_naca0012_35sect_20chord', 'none'),
    ('5DV_35S_60C', 5, 'rectangular_wing_naca0012_35sect_60chord', 'none'),
]

print("Starting Chordwise Refinement Study (startup-pinned to P-cores 0-11)...", flush=True)
for name, n_dv, mesh_name, payload in cases:
    t0 = time.time()
    print(f"\n--- Launching {name}: DVs={n_dv}, Mesh={mesh_name}, Payload={payload} ---", flush=True)
    cmd = (
        f"BWB_NUM_STATIONS={n_dv} "
        f"BWB_AERO_MESH_FILE={mesh_name} "
        f"GEONIC_PAYLOAD_MODE={payload} "
        f"taskset -c 0-11 /home/andrewfletcher/anaconda3/envs/py310/bin/python "
        f"examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py "
        f"> /tmp/diag_{name}.log 2>&1"
    )
    ret = os.system(cmd)
    dt = time.time() - t0
    if ret == 0:
        print(f"SUCCESS: {name} completed in {dt:.1f} s ({dt/3600:.2f} h)", flush=True)
    else:
        print(f"FAILED: {name} (exit code {ret}) in {dt:.1f} s. Check /tmp/diag_{name}.log", flush=True)
