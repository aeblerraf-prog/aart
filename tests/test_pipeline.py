import os
import sys
import pytest
import h5py
import numpy as np
import shutil

# Add the repository root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to import the modules to test them
import aart_func.lb_f as lb_f
import aart_func.raytracing_f as rt_f
# We import params to get the path variable to check outputs
from params import path as results_path, spin_case, i_case

@pytest.fixture
def setup_test_env():
    # Ensure Results directory exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    yield
    # Cleanup could go here, but checking artifacts is useful

def test_pipeline_low_res(setup_test_env):
    """
    Runs the pipeline with low resolution to verify connectivity.
    """

    # 1. Modify params in lb_f for low res
    orig_npointsS = lb_f.npointsS
    orig_dx0 = lb_f.dx0
    orig_dx1 = lb_f.dx1
    orig_dx2 = lb_f.dx2
    orig_limits = lb_f.limits

    try:
        lb_f.npointsS = 20
        lb_f.dx0 = 2.0
        lb_f.dx1 = 2.0
        lb_f.dx2 = 2.0
        lb_f.limits = 10

        # 2. Run Lensing Bands (using the library function directly)
        print("Running Lensing Bands...")
        lb_f.lb()

        expected_lb_file = f"{results_path}LensingBands_a_{lb_f.spin_case}_i_{lb_f.i_case}.h5"
        assert os.path.exists(expected_lb_file)

        # 3. Prepare data for Ray Tracing
        # Raytracing script reads from file, so we mimic that or call rt directly.
        # calling rt directly is better for unit testing logic.

        with h5py.File(expected_lb_file, 'r') as f:
            supergrid0 = f['grid0'][:]
            mask0 = f['mask0'][:]
            supergrid1 = f['grid1'][:]
            mask1 = f['mask1'][:]
            supergrid2 = f['grid2'][:]
            mask2 = f['mask2'][:]

        # 4. Run Ray Tracing
        print("Running Ray Tracing...")
        rt_f.rt(supergrid0, mask0, supergrid1, mask1, supergrid2, mask2)

        expected_rt_file = f"{results_path}Rays_a_{lb_f.spin_case}_i_{lb_f.i_case}.h5"
        assert os.path.exists(expected_rt_file)

        with h5py.File(expected_rt_file, 'r') as f:
             assert 'rs0' in f
             assert 't0' in f

    finally:
        # Restore
        lb_f.npointsS = orig_npointsS
        lb_f.dx0 = orig_dx0
        lb_f.dx1 = orig_dx1
        lb_f.dx2 = orig_dx2
        lb_f.limits = orig_limits
