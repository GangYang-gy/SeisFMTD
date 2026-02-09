# -------------------------------------------------------------------
# Prepare the data and SGT to conduct the inversion.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# Modified By: Gang Yang (gangy.yang@mail.utoronto.ca)
# -------------------------------------------------------------------

from pyCAPLunar import g_TAPER_SCALE, g_TAPER_RATE_SRF
from pyCAPLunar.DFilters import DFilter_sgt, DFilter_data
from pyCAPLunar.DCut import DCut_sgt_pnl, DCut_sgt_srf,DCut_data_pnl, DCut_data_srf
import numpy as np
from obspy.core.trace import Trace 


def prepare_sgt(sgt,df, p_freqmin, p_freqmax,
                s_freqmin, s_freqmax, 
                n_tp, n_p_length,
                n_ts, n_s_length, filter):
    '''
    Filter and cut SGT.

    :param sgt:         The sgt array for one station.
                        Data shape: [n_sample, n_dim, n_para]
    :param p_freqmin:   The low freq limit of the bandpass filter for the PNL.
    :param p_freqmax:   The high freq limit of the bandpass filter for the PNL.
    :param s_freqmin:   The low freq limit of the bandpass filter for the S/Surface.
    :param s_freqmax:   The high freq limit of the bandpass filter for the S/Surface.
    :param df:          The sampling rate in Hz.
    :param n_tp:        The computational arrival time of the Pnl phase in sample. INT.
    :param n_p_length:  The length of the cut Pnl phase. INT.
    :param n_ts:        The computational arrival time of the S/Surface phase in sample. INT.
    :param n_s_length:  The length of the cut S/Surface phase. INT.
    :return:            A list of the cut and filtered SGT data of phases.
                        Data shape: [[PNL_step, ndim, npara], [S/Surface_step, ndim, npara]
    '''
    
    if filter:
        sgt_pnl = DFilter_sgt(sgt, p_freqmin, p_freqmax, df)
        if p_freqmin == s_freqmin and p_freqmax == s_freqmax:
            sgt_srf = sgt_pnl.copy()
        else:
            sgt_srf = DFilter_sgt(sgt, s_freqmin, s_freqmax, df)
    else:
        sgt_pnl =  sgt
        sgt_srf = sgt

    pnl_sgt = DCut_sgt_pnl(sgt_pnl, n_tp, n_p_length)
    srf_sgt = DCut_sgt_srf(sgt_srf, n_ts, n_s_length)
    sgt_list = [pnl_sgt, srf_sgt]
   
    return sgt_list


def prepare_data(data, 
                df_data, df,
                p_freqmin, p_freqmax,
                s_freqmin, s_freqmax, 
                n_tp, n_p_length,
                n_ts, n_s_length, filter):               
    '''
    Preapre the data for single station.
    Only Filter.

    :param data:        The 3C waveform for one station. Data size: [3 * n_sample]
    :param p_freqmin:   The low freq limit of the bandpass filter for the PNL.
    :param p_freqmax:   The high freq limit of the bandpass filter for the PNL.
    :param s_freqmin:   The low freq limit of the bandpass filter for the S/Surface.
    :param s_freqmax:   The high freq limit of the bandpass filter for the S/Surface.
    :param df:          The sampling rate in Hz.
    :param n_tp:        The arrival time of p phase.
    :param n_ts:        The arrival tiem of S phase.
    :return:            The data list for one staiton [[PNL: 3 * n_sample], [S/Surface: 3 * n_sample]]
    '''

    if filter:
        data_pnl = DFilter_data(data, p_freqmin, p_freqmax, df_data, df)
        if p_freqmin == s_freqmin and p_freqmax == s_freqmax:
            data_srf = data_pnl.copy()
        else:
            data_srf = DFilter_data(data, s_freqmin, s_freqmax, df_data, df)
        df_data = df
    else:
        data_pnl = data
        data_srf = data
        
    pnl_data = DCut_data_pnl(data_pnl, n_tp, int(n_p_length*df_data/df))
    srf_data = DCut_data_srf(data_srf, n_ts, int(n_s_length*df_data/df))
  
    data_list = [pnl_data, srf_data]

    return data_list


def _prepare_N_station_sgt(sgt_n_stations, df, p_freqmin, p_freqmax,
                            s_freqmin, s_freqmax, 
                            n_tp_sgt, n_p_length, 
                            n_ts_sgt, n_s_length, filter=False):
    sgt_list_n_stations = []
    n_station = len(sgt_n_stations)
    for i in range(n_station):
        # sgt
        sgt_list_n_stations.append(prepare_sgt(sgt_n_stations[i],df, 
                                               p_freqmin, p_freqmax,
                                               s_freqmin, s_freqmax,   
                                               n_tp_sgt[i], n_p_length,
                                               n_ts_sgt[i], n_s_length, filter))
    return sgt_list_n_stations


def _prepare_N_station_data(data_n_stations, df_data, df, p_freqmin, p_freqmax,
                            s_freqmin, s_freqmax, 
                            n_tp_data, n_p_length,
                            n_ts_data, n_s_length, filter=False):
    data_list_n_stations = []
    n_station = len(data_n_stations)
    for i in range(n_station):
        # data
        data_list_n_stations.append(prepare_data(data_n_stations[i], 
                                                 df_data, df, 
                                                 p_freqmin, p_freqmax,
                                                 s_freqmin, s_freqmax, 
                                                 n_tp_data[i], n_p_length,
                                                 n_ts_data[i], n_s_length, filter))
    return data_list_n_stations


