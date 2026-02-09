#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full Moment Tensor Inversion using HMC - Example Script
========================================================

This example demonstrates how to use SeisFMTD for seismic full moment tensor
inversion using the Hamiltonian Monte Carlo (HMC) method.

Before running this script, you need to prepare:
1. SGT (Strain Green's Tensor) data for each station
2. Observed seismic waveforms
3. Station and source location information

Author: Gang Yang (gangy.yang@mail.utoronto.ca)
"""

import os
import sys
import time
import numpy as np

# Add SeisFMTD to path (modify to your installation path)
sys.path.insert(0, '/path/to/SeisFMTD')  # <-- YOUR PATH HERE

from SeisFMTD.pyCAPLunar import g_TAPER_SCALE, g_TAPER_RATE_SRF
from SeisFMTD.pyCAPSolvers.DHMC_linear import DHMC_MT
from SeisFMTD.MTTools.DMomentTensors import DMT_enz
from SeisFMTD.pyCAPLunar.DCAPUtils import mag2moment


# =============================================================================
# DATA PREPARATION FUNCTIONS (You need to implement these)
# =============================================================================

def load_sgt_data(station_ids, loc_stations, loc_src):
    """
    Load Strain Green's Tensor data for each station.
    
    You need to implement this function to load your SGT data.
    
    Parameters
    ----------
    station_ids : list of str
        List of station names, e.g., ['STA1', 'STA2', 'STA3']
    loc_stations : list of array
        Station locations [[lat, lon, depth], ...] in degrees and km
    loc_src : list
        Source location [lat, lon, depth] in degrees and km
    
    Returns
    -------
    sgt_list : list
        List of SGT data for each station. Each element is:
        [pnl_sgt, srf_sgt] where:
        - pnl_sgt: np.ndarray of shape (n_samples_pnl, 3, 6)
                   P-wave SGT in NEZ order, 6 MT components
        - srf_sgt: np.ndarray of shape (n_samples_srf, 3, 6)
                   S-wave SGT in NEZ order, 6 MT components
    
    n_tp_sgt : list of int
        P-wave arrival sample index in SGT for each station
    
    n_ts_sgt : list of int
        S-wave arrival sample index in SGT for each station
    """
    # Example structure:
    # sgt_list = []
    # for sta in station_ids:
    #     pnl_sgt = np.load(f'/your/sgt/path/{sta}_pnl.npy')  # shape: (n_samples, 3, 6)
    #     srf_sgt = np.load(f'/your/sgt/path/{sta}_srf.npy')  # shape: (n_samples, 3, 6)
    #     sgt_list.append([pnl_sgt, srf_sgt])
    
    raise NotImplementedError("Please implement load_sgt_data() for your dataset")


def load_observed_data(station_ids, loc_stations, loc_src):
    """
    Load observed seismic waveforms for each station.
    
    You need to implement this function to load your observed data.
    
    Parameters
    ----------
    station_ids : list of str
        List of station names
    loc_stations : list of array
        Station locations [[lat, lon, depth], ...]
    loc_src : list
        Source location [lat, lon, depth]
    
    Returns
    -------
    data_list : list
        List of observed data for each station. Each element is:
        [pnl_data, srf_data] where:
        - pnl_data: np.ndarray of shape (3, n_samples_pnl)
                    P-wave data in ENZ order
        - srf_data: np.ndarray of shape (3, n_samples_srf)
                    S-wave data in ENZ order
    
    n_tp_data : list of int
        P-wave arrival sample index in data for each station
    
    n_ts_data : list of int
        S-wave arrival sample index in data for each station
    """
    # Example structure:
    # data_list = []
    # for sta in station_ids:
    #     from obspy import read
    #     st = read(f'/your/data/path/{sta}*.sac')
    #     # Process and extract P and S windows
    #     pnl_data = np.array([st_E_pnl, st_N_pnl, st_Z_pnl])  # shape: (3, n_samples)
    #     srf_data = np.array([st_E_srf, st_N_srf, st_Z_srf])  # shape: (3, n_samples)
    #     data_list.append([pnl_data, srf_data])
    
    raise NotImplementedError("Please implement load_observed_data() for your dataset")


# =============================================================================
# MAIN INVERSION EXAMPLE
# =============================================================================

def run_example():
    """
    Example: Run HMC moment tensor inversion.
    """
    
    # =========================================================================
    # STEP 1: Define source and station information
    # =========================================================================
    
    # Source location [latitude (°), longitude (°), depth (km)]
    loc_src = [35.0, -97.5, 5.0]  # <-- YOUR SOURCE LOCATION
    
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
    # STEP 2: Load data (implement the functions above)
    # =========================================================================
    
    # Load SGT data
    # sgt_list: List of [pnl_sgt, srf_sgt] for each station
    #   - pnl_sgt shape: (n_samples, 3, 6) - NEZ components, 6 MT elements
    #   - srf_sgt shape: (n_samples, 3, 6) - NEZ components, 6 MT elements
    sgt_list, n_tp_sgt, n_ts_sgt = load_sgt_data(station_ids, loc_stations, loc_src)
    
    # Load observed waveforms
    # data_list: List of [pnl_data, srf_data] for each station
    #   - pnl_data shape: (3, n_samples) - ENZ components
    #   - srf_data shape: (3, n_samples) - ENZ components
    data_list, n_tp_data, n_ts_data = load_observed_data(station_ids, loc_stations, loc_src)
    
    # =========================================================================
    # STEP 3: Set timing parameters
    # =========================================================================
    
    # Sampling interval (seconds)
    dt = 0.5  # <-- YOUR SAMPLING INTERVAL
    
    # Maximum allowed time shifts for cross-correlation (in samples)
    n_tp_max_shift = 5   # P-wave max shift
    n_ts_max_shift = 10  # S-wave max shift
    
    # =========================================================================
    # STEP 4: Configure solver parameters
    # =========================================================================
    
    # Mass matrix inverse (controls step sizes in HMC)
    # Smaller values = larger steps, faster but less precise
    n_params = 6
    massinv = np.eye(n_params) / 100.0
    
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
    # STEP 5: Initialize the HMC solver
    # =========================================================================
    
    print("="*60)
    print("Initializing DHMC_MT Solver")
    print("="*60)
    print(f"Number of stations: {len(station_ids)}")
    print(f"Source location: {loc_src}")
    print(f"Magnitude: Mw {mag}")
    
    solver = DHMC_MT(
        # Required data inputs
        sgts_list=sgt_list,           # SGT data for all stations
        data_list=data_list,          # Observed waveforms for all stations
        
        # Source parameters
        mag=mag,                       # Magnitude for moment scaling
        massinv=massinv,               # Inverse mass matrix (6x6)
        loc_src=loc_src,               # Source location [lat, lon, depth]
        
        # Station parameters
        loc_stations=loc_stations,     # Station locations
        station_names=station_ids,     # Station names
        
        # Timing parameters (sample indices)
        n_tp_sgt=n_tp_sgt,             # P arrival index in SGT
        n_ts_sgt=n_ts_sgt,             # S arrival index in SGT
        n_tp_data=n_tp_data,           # P arrival index in data
        n_ts_data=n_ts_data,           # S arrival index in data
        n_tp_max_shift=n_tp_max_shift, # Max P-wave time shift (samples)
        n_ts_max_shift=n_ts_max_shift, # Max S-wave time shift (samples)
        dt=dt,                         # Sampling interval (seconds)
        
        # Phase configuration
        n_phase=5,                     # Number of phases (2 Pnl + 3 Srf)
        n_component=3,                 # Number of components (ENZ)
        
        # Weighting
        w_pnl=w_pnl,                   # P-wave weight
        w_srf=w_srf,                   # S-wave weight
        
        # Quality control
        reject_rate=reject_rate,
        misfit_threshold=misfit_threshold,
        cc_threshold=cc_threshold,
        
        # Tapering
        taper_scale=taper_scale,
        taper_rate_srf=taper_rate_srf,
    )
    
    # =========================================================================
    # STEP 6: Set initial moment tensor
    # =========================================================================
    
    # Initial guess in source parameters:
    # [strike (°), dip (°), rake (°), magnitude, delta (°), gamma (°)]
    # delta: colatitude on lune (90° = deviatoric)
    # gamma: longitude on lune (0° = double-couple)
    
    strike_init = 45.0   # Strike angle in degrees
    dip_init = 60.0      # Dip angle in degrees
    rake_init = 30.0     # Rake angle in degrees
    mag_init = mag       # Magnitude
    delta_init = 90.0    # Colatitude on lune (90° = deviatoric)
    gamma_init = 0.0     # Longitude on lune (0° = DC)
    
    # Convert to moment tensor (ENU convention)
    mt_initial = mag2moment(mag_init) * DMT_enz(
        np.deg2rad(strike_init),
        np.deg2rad(dip_init),
        np.deg2rad(rake_init),
        np.deg2rad(delta_init),
        np.deg2rad(gamma_init)
    )
    mt_initial[3:] = mt_initial[3:] / 2  # Adjust off-diagonal elements
    
    solver.set_q(mt_initial)
    
    # =========================================================================
    # STEP 7: Configure output directory
    # =========================================================================
    
    output_dir = './results/'  # <-- YOUR OUTPUT DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    solver.set_saving_dir(output_dir)
    
    # =========================================================================
    # STEP 8: Run HMC sampling
    # =========================================================================
    
    print("\n" + "="*60)
    print("Running HMC with Dual-Averaging")
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
    # STEP 9: Analyze results
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
    
    # Convert mean to Tape parameters
    from SeisFMTD.pyCAPSolvers.DHMC_linear import convert_to_tapeq
    tape = convert_to_tapeq(mean)
    print(f"\nSource parameters:")
    print(f"  Strike: {tape[0]:.1f}°")
    print(f"  Dip:    {tape[1]:.1f}°")
    print(f"  Rake:   {tape[2]:.1f}°")
    print(f"  Mw:     {tape[3]:.2f}")
    print(f"  Delta:  {tape[4]:.1f}°")
    print(f"  Gamma:  {tape[5]:.1f}°")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    run_example()
