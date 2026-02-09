#!/usr/bin/env python3
"""
Written in March-April 2018
updated August 2018
"""

import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
import obspy.taup as taup
from obspy import read
from obspy.core.stream import Stream
import numpy as np
import os

def m_to_deg(distance_in_m):
    return kilometers2degrees(distance_in_m/1000., radius=6371.)

def get_arrival(arrivals, phase):
    phases = []
    for arrival in arrivals:
        phases += [arrival.phase.name]

    if phase not in phases:
        raise Exception("Phase not found")

    arrival = arrivals[phases.index(phase)]
    return arrival.time


def process_data(loc_src, wind_p, wind_surf,tp_max_shift,ts_max_shift):

#====================================initialization===================================

    id_list_n_stations = []
    loc_stations = []
    data_list_n_stations = [] 
    n_tp_data = []
    n_ts_data = []
    origin_depth_in_km =np.abs(loc_src[2])
#==================================== replace with forward data===================================
    stream = read('./syn_sac/*')

    stream_p = stream.copy()
    stream_s = stream.copy()

#==================================== calculate arrivals===================================
    taup_model = 'AK135'
    _taup = taup.TauPyModel(taup_model)
    
    p_picks = []
    s_picks = []
    print('the number of stream is: ', int(len(stream)/3))
    for i in range(int(len(stream)/3)):
      
        id_list_n_stations.append(stream[i*3].stats.station)
        stla = stream[i*3].stats.sac.stla
        stlo = stream[i*3].stats.sac.stlo
        stdp = 0
        loc_stations.append(np.array([stla, stlo, stdp]))
        # calculate phase arrivals
        distance_in_m, _, _ = gps2dist_azimuth(loc_src[0], loc_src[1], stla, stlo)
        arrivals = _taup.get_travel_times(origin_depth_in_km,
                                        m_to_deg(distance_in_m),
                                        phase_list=['p', 's', 'P', 'S'])
        try:
            P_pick = get_arrival(arrivals, 'p')
        except:
            P_pick = get_arrival(arrivals, 'P')
        try:
            S_pick = get_arrival(arrivals, 's')
        except:
            S_pick = get_arrival(arrivals, 'S')

#==================================== pnl window & surface window in ENZ order===================================
        dt = stream[i*3].stats.delta
        df = 1/dt

        before_id = stream[i*3].stats.network+'.'+stream[i*3].stats.station+'.'+stream[i*3].stats.location+'.'+stream[i*3].stats.channel[:2]
        Ep_comp = stream_p.select(id=before_id+'1')
        Es_comp = stream_s.select(id=before_id+'1')
   
        if len(Ep_comp) != 0:
            Np_comp = stream_p.select(id=before_id+'2')
            Ns_comp = stream_s.select(id=before_id+'2')
        else:
            Ep_comp = stream_p.select(id=before_id+'E')
            Np_comp = stream_p.select(id=before_id+'N')
            Es_comp = stream_s.select(id=before_id+'E')
            Ns_comp = stream_s.select(id=before_id+'N')

        Zp_comp = stream_p.select(id=before_id+'Z')
        Zs_comp = stream_s.select(id=before_id+'Z')
        
        n_p_fill = int(df * tp_max_shift)
        n_s_fill = int(df * ts_max_shift)
        
   
        '''
        if int((P_pick-0.4*wind_p)/dt) > 0 and int((S_pick-0.3*wind_surf)/dt)>0:
            pnl_N = np.hstack([np.zeros(n_p_fill),Np_comp[0].data[int((P_pick-0.4*wind_p)/dt):int((P_pick+0.6*wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_E = np.hstack([np.zeros(n_p_fill),Ep_comp[0].data[int((P_pick-0.4*wind_p)/dt):int((P_pick+0.6*wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_Z = np.hstack([np.zeros(n_p_fill),Zp_comp[0].data[int((P_pick-0.4*wind_p)/dt):int((P_pick+0.6*wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_data = np.array([pnl_E, pnl_N, pnl_Z])
            n_tp_data.append(n_p_fill+int(0.4*wind_p/dt))

            srf_N = np.hstack([np.zeros(n_s_fill), Ns_comp[0].data[int((S_pick-0.3*wind_surf)/dt):int((S_pick+0.7*wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_E = np.hstack([np.zeros(n_s_fill), Es_comp[0].data[int((S_pick-0.3*wind_surf)/dt):int((S_pick+0.7*wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_Z = np.hstack([np.zeros(n_s_fill), Zs_comp[0].data[int((S_pick-0.3*wind_surf)/dt):int((S_pick+0.7*wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_data = np.array([srf_E, srf_N, srf_Z])
            n_ts_data.append(n_s_fill+int(0.3*wind_surf/dt))
        elif int((P_pick-0.4*wind_p)/dt) < 0 and int((S_pick-0.3*wind_surf)/dt)>0:
            pnl_N = np.hstack([np.zeros(n_p_fill),Np_comp[0].data[int(P_pick/dt):int((P_pick+wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_E = np.hstack([np.zeros(n_p_fill),Ep_comp[0].data[int(P_pick/dt):int((P_pick+wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_Z = np.hstack([np.zeros(n_p_fill),Zp_comp[0].data[int(P_pick/dt):int((P_pick+wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_data = np.array([pnl_E, pnl_N, pnl_Z])
            n_tp_data.append(n_p_fill)

            srf_N = np.hstack([np.zeros(n_s_fill), Ns_comp[0].data[int((S_pick-0.3*wind_surf)/dt):int((S_pick+0.7*wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_E = np.hstack([np.zeros(n_s_fill), Es_comp[0].data[int((S_pick-0.3*wind_surf)/dt):int((S_pick+0.7*wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_Z = np.hstack([np.zeros(n_s_fill), Zs_comp[0].data[int((S_pick-0.3*wind_surf)/dt):int((S_pick+0.7*wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_data = np.array([srf_E, srf_N, srf_Z])
            n_ts_data.append(n_s_fill+int(0.3*wind_surf/dt))
        elif int((P_pick-0.4*wind_p)/dt) > 0 and int((S_pick-0.3*wind_surf)/dt) < 0:
            pnl_N = np.hstack([np.zeros(n_p_fill),Np_comp[0].data[int((P_pick-0.4*wind_p)/dt):int((P_pick+0.6*wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_E = np.hstack([np.zeros(n_p_fill),Ep_comp[0].data[int((P_pick-0.4*wind_p)/dt):int((P_pick+0.6*wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_Z = np.hstack([np.zeros(n_p_fill),Zp_comp[0].data[int((P_pick-0.4*wind_p)/dt):int((P_pick+0.6*wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_data = np.array([pnl_E, pnl_N, pnl_Z])
            n_tp_data.append(n_p_fill+int(0.4*wind_p/dt))

            srf_N = np.hstack([np.zeros(n_s_fill), Ns_comp[0].data[int(S_pick/dt):int((S_pick+wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_E = np.hstack([np.zeros(n_s_fill), Es_comp[0].data[int(S_pick/dt):int((S_pick+wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_Z = np.hstack([np.zeros(n_s_fill), Zs_comp[0].data[int(S_pick/dt):int((S_pick+wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_data = np.array([srf_E, srf_N, srf_Z])
            n_ts_data.append(n_s_fill)
        else:
            pnl_N = np.hstack([np.zeros(n_p_fill),Np_comp[0].data[int(P_pick/dt):int((P_pick+wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_E = np.hstack([np.zeros(n_p_fill),Ep_comp[0].data[int(P_pick/dt):int((P_pick+wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_Z = np.hstack([np.zeros(n_p_fill),Zp_comp[0].data[int(P_pick/dt):int((P_pick+wind_p)/dt)], np.zeros(n_p_fill)])
            pnl_data = np.array([pnl_E, pnl_N, pnl_Z])
            n_tp_data.append(n_p_fill)

            srf_N = np.hstack([np.zeros(n_s_fill), Ns_comp[0].data[int(S_pick/dt):int((S_pick+wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_E = np.hstack([np.zeros(n_s_fill), Es_comp[0].data[int(S_pick/dt):int((S_pick+wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_Z = np.hstack([np.zeros(n_s_fill), Zs_comp[0].data[int(S_pick/dt):int((S_pick+wind_surf)/dt)], np.zeros(n_s_fill)])
            srf_data = np.array([srf_E, srf_N, srf_Z])
            n_ts_data.append(n_s_fill)
        '''
        pnl_N = Np_comp[0].data[:200]
        pnl_E = Ep_comp[0].data[:200]
        pnl_Z = Zp_comp[0].data[:200]
        pnl_data = np.array([pnl_E, pnl_N, pnl_Z])
        n_tp_data.append(0)

        srf_N = Ns_comp[0].data[:200]
        srf_E = Es_comp[0].data[:200]
        srf_Z = Zs_comp[0].data[:200]
        srf_data = np.array([srf_E, srf_N, srf_Z])
        n_ts_data.append(0)

        data = [pnl_data, srf_data]
        
        #print(i,id_list_n_stations[i],len(pnl_data[0]), len(srf_data[0]))
        data_list_n_stations.append(data)
        
        p_picks.append(P_pick)
        s_picks.append(S_pick)
    
    return id_list_n_stations, loc_stations, data_list_n_stations,p_picks,s_picks, n_tp_data, n_ts_data

