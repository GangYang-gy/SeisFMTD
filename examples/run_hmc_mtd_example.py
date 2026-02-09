#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Moment Tensor + Depth Inversion using HMC - Example Script
================================================================

This example demonstrates how to use SeisFMTD for seismic full moment tensor
AND source depth inversion using the Hamiltonian Monte Carlo (HMC) method.

The key difference from `run_hmc_example.py`:
- Uses DHMC_MTD instead of DHMC_MT
- Inverts for 7 parameters: 6 MT components + depth
- Requires SGT manager to query Green's functions at different depths

Before running this script, you need to prepare:
1. SGT database with seisgen/DSGTMgr
2. Pre-computed 3D velocity model database
3. Observed seismic waveforms
4. Station and source location information

Author: Gang Yang (gangy.yang@mail.utoronto.ca)
"""

import os
import sys
import time
import numpy as np

# Add SeisFMTD to path (modify to your installation path)
sys.path.insert(0, '/path/to/SeisFMTD')  # <-- YOUR PATH HERE
sys.path.insert(0, '/path/to/seisgen')   # <-- YOUR SEISGEN PATH HERE

from SeisFMTD.pyCAPLunar import g_TAPER_SCALE, g_TAPER_RATE_SRF
from SeisFMTD.pyCAPSolvers.DHMC_mtd_linear import DHMC_MTD
from SeisFMTD.MTTools.DMomentTensors import DMT_enz
from SeisFMTD.pyCAPLunar.DCAPUtils import mag2moment
from seisgen.DSGTMgr import DSGTMgr


# =============================================================================
# SGT MANAGER SETUP (You need to configure these paths)
# =============================================================================

def setup_sgt_manager():
    """
    Initialize the SGT (Strain Green's Tensor) manager.
    
    You need to configure these paths for your SGT database.
    The SGT manager handles real-time interpolation of Green's functions
    at different source depths.
    
    Returns
    -------
    sgtMgr : DSGTMgr
        Initialized SGT manager object
    sgt_database_dir : str
        Path to the SGT database directory
    """
    # SGT database paths - MODIFY THESE TO YOUR PATHS
    sgt_database_dir = '/path/to/sgt_database/'       # <-- YOUR SGT DATABASE
    model_database_dir = '/path/to/model_database/'   # <-- YOUR MODEL DATABASE
    point_cloud_file = '/path/to/point_cloud.hdf5'    # <-- YOUR POINT CLOUD FILE
    
    # Initialize SGT manager
    sgtMgr = DSGTMgr(sgt_database_dir, model_database_dir, point_cloud_file)
    
    return sgtMgr, sgt_database_dir


# =============================================================================
# DATA LOADING (You need to implement this)
# =============================================================================

def load_observed_data(station_ids):
    """
    Load observed seismic waveforms for each station.
    
    Parameters
    ----------
    station_ids : list of str
        List of station names
    
    Returns
    -------
    data_n_stations : list
        Raw observed data for each station.
        Each element is a 2D array of shape (3, n_samples) in ENZ order.
        This is the FULL waveform, not cut into P and S windows.
        The DHMC_MTD class will handle windowing internally.
    """
    # Example structure:
    # data_n_stations = []
    # for sta in station_ids:
    #     from obspy import read
    #     st = read(f'/your/data/path/{sta}*.sac')
    #     # Stack ENZ components into array
    #     data = np.array([st.select(component='E')[0].data,
    #                      st.select(component='N')[0].data,
    #                      st.select(component='Z')[0].data])
    #     data_n_stations.append(data)
    
    raise NotImplementedError("Please implement load_observed_data() for your dataset")


# =============================================================================
# MAIN INVERSION EXAMPLE
# =============================================================================

def run_example():
    """
    Example: Run HMC moment tensor + depth inversion.
    """
    
    # =========================================================================
    # STEP 1: Define source and station information
    # =========================================================================
    
    # Source location [latitude (°), longitude (°), initial depth (km)]
    # Note: depth is positive downward
    loc_src = [35.0, -97.5, 5.0]  # <-- YOUR SOURCE LOCATION (initial guess for depth)
    
    # Magnitude estimate (used for scaling)
    mag = 4.0  # <-- YOUR MAGNITUDE ESTIMATE
    
    # Station information
    station_ids = ['STA1', 'STA2', 'STA3']  # <-- YOUR STATION NAMES
    
    # Station locations: [[lat, lon, depth], ...] for each station
    loc_stations = [
        [35.1, -97.4, 0.0],  # STA1
        [35.2, -97.6, 0.0],  # STA2
        [34.9, -97.3, 0.0],  # STA3
    ]  # <-- YOUR STATION LOCATIONS
    
    # =========================================================================
    # STEP 2: Initialize SGT manager
    # =========================================================================
    
    print("="*60)
    print("Initializing SGT Manager")
    print("="*60)
    
    sgtMgr, sgt_database_dir = setup_sgt_manager()
    
    # =========================================================================
    # STEP 3: Load observed data
    # =========================================================================
    
    # Load raw waveform data (full trace, not windowed)
    # data_n_stations: List of (3, n_samples) arrays in ENZ order
    data_n_stations = load_observed_data(station_ids)
    
    # =========================================================================
    # STEP 4: Set timing and filter parameters
    # =========================================================================
    
    # Sampling rates
    df = 2.0        # Target sampling rate (Hz) for SGT
    df_data = 2.0   # Sampling rate of observed data (Hz)
    
    # Bandpass filter parameters (Hz)
    p_freqmin, p_freqmax = 1/30, 1/2   # P-wave: 2-30s period
    s_freqmin, s_freqmax = 1/30, 1/5   # S-wave: 5-30s period
    
    # Phase window lengths (seconds)
    n_p_length = 12   # P-wave window length
    n_s_length = 40   # S-wave window length
    
    # P and S arrival sample indices in DATA (for each station)
    # You need to calculate these based on travel time predictions
    n_tp_data = [20, 22, 18]  # <-- YOUR P ARRIVAL INDICES
    n_ts_data = [40, 45, 38]  # <-- YOUR S ARRIVAL INDICES
    
    # Maximum allowed time shifts (samples)
    n_tp_max_shift = 4   # P-wave max shift
    n_ts_max_shift = 10  # S-wave max shift
    
    # =========================================================================
    # STEP 5: Configure solver parameters
    # =========================================================================
    
    # Mass matrix inverse (controls step sizes in HMC)
    # Note: 7 parameters for MT + depth
    n_params = 7
    massinv = np.eye(n_params)
    massinv[:6, :6] = np.eye(6) / 100.0  # MT components
    massinv[6, 6] = 1.0 / 10.0           # Depth (larger step)
    
    # Phase weighting
    w_pnl = 0.5   # Weight for P-wave (body wave)
    w_srf = 1.0   # Weight for S-wave (surface wave)
    
    # Quality control thresholds
    misfit_threshold = 5.0    # Maximum acceptable misfit
    cc_threshold = 0.2        # Minimum cross-correlation coefficient
    reject_rate = 1.0         # Rejection rate for bad fits
    
    # Tapering parameters
    taper_scale = g_TAPER_SCALE        # Default: -0.4
    taper_rate_srf = g_TAPER_RATE_SRF  # Default: 0.0
    
    # =========================================================================
    # STEP 6: Initialize the HMC solver
    # =========================================================================
    
    print("\n" + "="*60)
    print("Initializing DHMC_MTD Solver (Moment Tensor + Depth)")
    print("="*60)
    print(f"Number of stations: {len(station_ids)}")
    print(f"Source location: {loc_src}")
    print(f"Magnitude: Mw {mag}")
    print(f"Number of parameters: {n_params} (6 MT + 1 depth)")
    
    solver = DHMC_MTD(
        # SGT manager and database
        sgtMgr=sgtMgr,                 # SGT manager instance
        sgt_database_dir=sgt_database_dir,  # SGT database path
        
        # Source parameters
        mag=mag,                        # Magnitude for moment scaling
        
        # Station parameters
        station_names=station_ids,      # Station names
        
        # Sampling and filtering
        df=df,                          # Target sampling rate (Hz)
        df_data=df_data,                # Data sampling rate (Hz)
        p_freqmin=p_freqmin,            # P-wave filter low
        p_freqmax=p_freqmax,            # P-wave filter high
        s_freqmin=s_freqmin,            # S-wave filter low
        s_freqmax=s_freqmax,            # S-wave filter high
        
        # Phase window lengths (seconds)
        n_p_length=n_p_length,          # P-wave window length
        n_s_length=n_s_length,          # S-wave window length
        
        # Observed data
        data_n_stations=data_n_stations,  # Raw observed waveforms
        loc_src=loc_src,                  # Source location [lat, lon, depth]
        loc_stations=loc_stations,        # Station locations
        
        # Timing parameters
        n_tp_data=n_tp_data,            # P arrival index in data
        n_ts_data=n_ts_data,            # S arrival index in data
        n_tp_max_shift=n_tp_max_shift,  # Max P-wave time shift (samples)
        n_ts_max_shift=n_ts_max_shift,  # Max S-wave time shift (samples)
        
        # Mass matrix
        massinv=massinv,                # Inverse mass matrix (7x7)
        
        # Phase configuration
        n_phase=5,                      # Number of phases (2 Pnl + 3 Srf)
        n_component=3,                  # Number of components (ENZ)
        
        # Weighting
        w_pnl=w_pnl,                    # P-wave weight
        w_srf=w_srf,                    # S-wave weight
        
        # Quality control
        reject_rate=reject_rate,
        misfit_threshold=misfit_threshold,
        cc_threshold=cc_threshold,
        
        # Tapering
        taper_scale=taper_scale,
        taper_rate_srf=taper_rate_srf,
        
        # Whether to apply bandpass filter internally
        filter=False,  # Set True if data not pre-filtered
    )
    
    # =========================================================================
    # STEP 7: Set initial moment tensor + depth
    # =========================================================================
    
    # Initial guess in source parameters:
    # [strike (°), dip (°), rake (°), magnitude, delta (°), gamma (°)]
    # + depth (km)
    
    strike_init = 45.0     # Strike angle in degrees
    dip_init = 60.0        # Dip angle in degrees
    rake_init = 30.0       # Rake angle in degrees
    mag_init = mag         # Magnitude
    delta_init = 90.0      # Colatitude on lune (90° = deviatoric)
    gamma_init = 0.0       # Longitude on lune (0° = DC)
    depth_init = loc_src[2]  # Initial depth (km)
    
    # Convert to moment tensor (ENU convention)
    mt_initial = mag2moment(mag_init) * DMT_enz(
        np.deg2rad(strike_init),
        np.deg2rad(dip_init),
        np.deg2rad(rake_init),
        np.deg2rad(delta_init),
        np.deg2rad(gamma_init)
    )
    mt_initial[3:] = mt_initial[3:] / 2  # Adjust off-diagonal elements
    
    # Combine MT + depth into initial state vector (7 parameters)
    q_initial = np.hstack([mt_initial, depth_init])
    
    print(f"\nInitial depth: {depth_init} km")
    solver.set_q(q_initial)
    
    # =========================================================================
    # STEP 8: Configure output directory
    # =========================================================================
    
    output_dir = './results_mtd/'  # <-- YOUR OUTPUT DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    solver.set_saving_dir(output_dir)
    
    # =========================================================================
    # STEP 9: Run HMC sampling
    # =========================================================================
    
    print("\n" + "="*60)
    print("Running HMC with Dual-Averaging (MT + Depth)")
    print("="*60)
    
    # Configure Dual-Averaging parameters
    solver.set_hmc_da_params(
        target_acceptance=0.65,   # Target acceptance rate
        warmup_fraction=0.4,      # Fraction for warmup (step-size tuning)
        path_length=5,            # Integration path length
    )
    
    # Run sampling
    n_samples = 500  # Number of posterior samples
    t_start = time.time()
    
    samples, mean, misfits = solver.hmc_da(n_samples)
    
    elapsed = time.time() - t_start
    
    # =========================================================================
    # STEP 10: Analyze results
    # =========================================================================
    
    print("\n" + "="*60)
    print("Inversion Results")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Acceptance rate: {solver.acceptance_rate:.2%}")
    print(f"Adapted epsilon: {solver.epsilon:.6f}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nMean moment tensor (ENU):")
    print(f"  Mee = {mean[0]:.4e}")
    print(f"  Mnn = {mean[1]:.4e}")
    print(f"  Muu = {mean[2]:.4e}")
    print(f"  Men = {mean[3]:.4e}")
    print(f"  Meu = {mean[4]:.4e}")
    print(f"  Mnu = {mean[5]:.4e}")
    print(f"\nMean depth: {mean[6]:.2f} km")
    
    # Convert mean MT to Tape parameters
    from SeisFMTD.pyCAPSolvers.DHMC_mtd_linear import convert_to_tapeq
    tape = convert_to_tapeq(mean[:-1])  # Exclude depth
    print(f"\nSource parameters:")
    print(f"  Strike: {tape[0]:.1f}°")
    print(f"  Dip:    {tape[1]:.1f}°")
    print(f"  Rake:   {tape[2]:.1f}°")
    print(f"  Mw:     {tape[3]:.2f}")
    print(f"  Delta:  {tape[4]:.1f}°")
    print(f"  Gamma:  {tape[5]:.1f}°")
    print(f"  Depth:  {mean[6]:.2f} km")
    
    # Depth uncertainty from posterior
    depth_samples = samples[:, 6]
    depth_std = np.std(depth_samples)
    print(f"\nDepth uncertainty (1-sigma): ±{depth_std:.2f} km")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    run_example()
