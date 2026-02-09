# -------------------------------------------------------------------
# Try HMC inversion code.
#
# Author: Gang Yang
# Email: gangy.yang@mail.utoronto.ca
# -------------------------------------------------------------------

import sys
sys.path.insert(1,'/scratch/gang97/01_workshop/software/HMC_package/SeisFMTD')
sys.path.insert(1,'/scratch/gang97/01_workshop/software/seisgen')
from pyCAPLunar import g_TAPER_SCALE, g_TAPER_RATE_SRF
from pyCAPSolvers.DHMC_linear import DHMC_MT
from MTTools.DMomentTensors import DMT_enz
from pyCAPLunar.DCAPUtils import mag2moment
import numpy as np
import pickle
from data.process_data_to_sac import process_data
import os
import time
from obspy import UTCDateTime

def RTP_to_DENZ(mt):  #USE2ENU convention, rather than NED convention!!!
    new_mt = np.zeros_like(mt)
    new_mt[0]=mt[2]
    new_mt[1]=mt[1]
    new_mt[2]=mt[0]
    new_mt[3]=-1*mt[5]
    new_mt[4]=mt[4]
    new_mt[5]=-1*mt[3]
    new_mt[3:] = 2.0 * new_mt[3:]
    return new_mt

def get_phase_window(mag):
    if mag >= 3 and mag < 3.5:
        pnl_length = 8
        srf_length = 30
    elif mag >= 3.5 and mag < 4:
        pnl_length = 10
        srf_length = 35
    if mag >= 4 and mag < 4.5:
        pnl_length = 12
        srf_length = 40
    if mag >= 4.5 and mag < 5:
        pnl_length = 15
        srf_length = 45
    if mag >= 5 and mag < 5.5:
        pnl_length = 18
        srf_length = 50

    return pnl_length, srf_length

np.random.seed(1)
def inversion():

    Nq = 6
    massinv = np.eye(Nq)
    massinv[0,0] = 1.0/1e1
    massinv[1,1] = 1.0/1e1
    massinv[2,2] = 1.0/1e1
    massinv[3,3] = 1.0/1e1
    massinv[4,4] = 1.0/1e1
    massinv[5,5] = 1.0/1e1

    # get data and stations input   
    evt_id = '201408191241'
    origin = UTCDateTime("2014-08-19T12:41:00")
    loc_src = [35.82, -97.47, -3.5] 
    mag = 4.3
    strike = 300
    dip = 70
    rake = -10
    colatitude = 90
    longitude = 0

    # data and parameters to be set
    df = 2
    tp_max_shift = 0
    ts_max_shift = 0
    dt = 1/df

    n_tp_max_shift= int(tp_max_shift/dt)
    n_ts_max_shift = int(ts_max_shift/dt)
    
    pnl_length, srf_length = get_phase_window(mag)


    # ======================================generate data & sgt==========================================
    
    # process data

    id_list_n_stations, loc_stations, data_list_n_stations, p_picks, s_picks, n_tp_data, n_ts_data \
    = process_data(loc_src, pnl_length, srf_length, tp_max_shift,ts_max_shift)
          
    # generate sgt
    with open("sgt_stream.pkl", "rb") as f:
        data = pickle.load(f)

    name_stations = data["id_list_n_stations"]
    sgt_list_n_stations = data["sgt_list_n_stations"]
    n_tp_sgt_phase = data["n_tp_syn"]
    n_ts_sgt_phase = data["n_ts_syn"]

    # -----------------------------
    # 3. Find station indices where SGT exists
    # -----------------------------
    # Only keep stations from id_list_n_stations that have SGT data
    idx_exist_sgt = [i for i, sta in enumerate(id_list_n_stations) if sta in name_stations]

    # Find corresponding indices in original SGT data
    sgt_idx = [name_stations.index(id_list_n_stations[i]) for i in idx_exist_sgt]

    # -----------------------------
    # 4. Filter SGT and related data synchronously
    # -----------------------------
    filtered_id_list = [id_list_n_stations[i] for i in idx_exist_sgt]
    filtered_loc_stations = [loc_stations[i] for i in idx_exist_sgt]
    filtered_data_list = [data_list_n_stations[i] for i in idx_exist_sgt]
    filtered_p_picks = [p_picks[i] for i in idx_exist_sgt]
    filtered_s_picks = [s_picks[i] for i in idx_exist_sgt]
    filtered_n_tp_data = [n_tp_data[i] for i in idx_exist_sgt]
    filtered_n_ts_data = [n_ts_data[i] for i in idx_exist_sgt]

    filtered_sgt_list = [sgt_list_n_stations[i] for i in sgt_idx]
    filtered_tp_phase = [n_tp_sgt_phase[i] for i in sgt_idx]
    filtered_ts_phase = [n_ts_sgt_phase[i] for i in sgt_idx]

    trimmed_sgt_list = [
    [sgt[:200, :, :] for sgt in station_data] 
    for station_data in filtered_sgt_list       
    ]
#=======================================set parameters==========================================
    # data and parameters to be set
    n_tp_max_shift= int(tp_max_shift/dt)
    n_ts_max_shift = int(ts_max_shift/dt)
    
    # for HMC inversion
    misfit_threshold = 5.0
    reject_rate = 1.0
    cc_threshold = 0.2

    taper_scale = g_TAPER_SCALE  # must LESS than 0.
    taper_rate_srf = g_TAPER_RATE_SRF  # between [0, 1)
    w_pnl = 1.0
    w_srf = 1.0

    evt_dir = './results/'
    if not os.path.exists(evt_dir):
        os.makedirs(evt_dir, exist_ok=True)

    t0 = time.time()
    solver = DHMC_MT(trimmed_sgt_list, filtered_data_list,
                     mag, massinv,
                     loc_src, filtered_loc_stations, filtered_id_list,
                     filtered_tp_phase, filtered_ts_phase,
                     filtered_n_tp_data, filtered_n_ts_data,
                     n_tp_max_shift, n_ts_max_shift, 
                     dt, n_phase=5, n_component=3,
                     w_pnl=w_pnl, w_srf=w_srf,
                     misfit_threshold=5.0,
                     reject_rate=reject_rate,
                     taper_rate_srf=taper_rate_srf,
                     taper_scale=taper_scale,
                     cc_threshold=cc_threshold)


# ======================================inital model=========================================
    # initial solution.
    q = np.array([strike+10, dip+10, rake-10, mag+0.2, colatitude+5, longitude+5])   
    #q = np.array([strike, dip, rake, mag, colatitude, longitude])   
    mt0 = mag2moment(q[3]) * DMT_enz(np.deg2rad(q[0]), np.deg2rad(q[1]),
                                             np.deg2rad(q[2]), np.deg2rad(q[4]),
                                             np.deg2rad(q[5])) 

    mt0[3:] = mt0[3:]/2
    solver.set_q(mt0)
    solver.set_saving_dir(evt_dir)

# ======================================sampling==========================================
    
    epsilon = 0.05 #0.025
    nstep = None
    n_sample = 1000
    solver.set_hmc_da_params(warmup_fraction=0.4)
    samples,_,_ = solver.hmc_da(n_sample)

    para_file = evt_dir + str('_paras_') + 'samples_' + str(n_sample) + str('.txt')
    with open(para_file, 'w') as f:
        f.write('misfit threshold = {}\n'.format(misfit_threshold))
        f.write('reject rate = {}\n'.format(reject_rate))
        f.write('CC threshold = {}\n'.format(cc_threshold))
        f.write('taper scale = {}\n'.format(taper_scale))
        f.write('taper rate = {}\n'.format(taper_rate_srf))
        f.write('weight (PNL) = {}\n'.format(w_pnl))
        f.write('weight (S/Surface) = {}\n'.format(w_srf))
        f.write('\nSolver\n')
        f.write('Epsilon = {}\n'.format(solver.epsilon))
        f.write('Step = {}\n'.format(nstep))
        f.write('sigma_d = {}\n'.format(solver.sigma_d))
        f.write('# warmup sample = {}\n'.format(solver.M_adapt))
        f.write('# of sample = {}\n'.format(n_sample))
        f.write('accept_rate = {}\n'.format(solver.acceptance_rate))
        f.write("\nTime cost={}\n".format(np.round(time.time() - t0, 2)))

        print("End sampling")
    

if __name__ == '__main__':
    inversion()
