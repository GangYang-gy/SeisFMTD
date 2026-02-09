# -------------------------------------------------------------------
# Calculating the waveform misfit
# by using Cut-and-Paste and Tape's method.
#
# Ref:
# [1] TAPE Walter, TAPE Carl., 2015
# A uniform parametrization of moment tensors.
# GJI, 2015, 202(3): 2074–2081.
#
# [2] ZHU Lupei, HELMBERGER Donald V., 1996.
# Advancement in source estimation techniques using broadband regional seismograms.
# BSSA, 86(5): 1634–1641.
#
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# Modified By: Gang Yang (gangy.yang@mail.utoronto.ca)
# -------------------------------------------------------------------

from pyCAPLunar import g_TAPER_SCALE, g_TAPER_RATE_SRF
from pyCAPLunar.DSyn import DSynRotateRTZ, DENZ2RTZ
from pyCAPLunar.DPaste import DPaste, DPaste_shift_syn
from obspy.geodetics.base import gps2dist_azimuth

import numpy as np


class DCAP():
    '''Moment tensor inversion using the Cut-And-Paste method.'''

    def __init__(self, sgts_list, data_list,
                 loc_src, loc_stations,station_names,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=2.0,
                 reject_rate=1.0,
                 taper_scale=g_TAPER_SCALE, taper_rate_srf=g_TAPER_RATE_SRF,
                 cc_threshold=0.4,
                 amplitude_ratio_threshold=2.0):

        self.dt = dt
        self.fn = 1.0 / dt
        self.loc_src = np.zeros(3)
        self.loc_stations = loc_stations
        self.n_tp_sgt = np.asarray(n_tp_sgt).astype(int)
        self.n_ts_sgt = np.asarray(n_ts_sgt).astype(int)
        self.n_tp_data = np.asarray(n_tp_data).astype(int)
        self.n_ts_data = np.asarray(n_ts_data).astype(int)
        self.n_tp_max_shift = int(n_tp_max_shift)
        self.n_ts_max_shift = int(n_ts_max_shift)

        # constant variable
        self.n_phase = n_phase  # PNL in R and Z, and S/Surface in R, T and Z components.
        self.n_component = n_component  # 3, R-T-Z or E-N-Z component.
        self.n_station = len(sgts_list)
        self.n_segment = self.n_station * self.n_phase
        self.reject_rate = reject_rate
        self.misfit_threshold = misfit_threshold
        if np.fabs(cc_threshold) < 1:
            self.cc_threshold = np.fabs(cc_threshold)
        else:
            self.cc_threshold = np.fabs(cc_threshold) / 100.0
        self.MAX_MISFIT = 9999.0
        self.amp_ratio_threshold = amplitude_ratio_threshold

        # initial container
        self.back_azimuth_array = np.zeros(self.n_station)
        self.distance_array = np.zeros(self.n_station)
        self.sgt_pnl = []
        self.sgt_srf = []
        self.data_pnl_rtz = []
        self.data_srf_rtz = []

        # weighting factors
        self.w_pnl = w_pnl
        self.w_srf = w_srf

        # weight by distance, distance scale
        # Using Zhu, Lupei (1996)'s method
        self.w_pnl_distance_scale = 1.13
        self.w_Love_distance_scale = 0.55
        self.w_Rayleigh_distance_scale = 0.74

        # initialize.
        self.update_azimuth(loc_src)
        self.update_sgt(sgts_list)
        self.update_data(data_list)
        #self.create_weights()

        # create the coefficient for accelerating the inversion.
        self.TAPER_SCALE = taper_scale  # should be LESS than 0.
        self.taper_rate_srf = taper_rate_srf  # if taper_rate_srf = 1.0, the s/surface will be discard.
        self.taper_coeff = np.exp(self.TAPER_SCALE * np.arange(np.max(self.n_syn_lens_srf)))

        self.station_names = station_names

    def update_azimuth(self, loc_src):
        '''Update azimuth and data component in R-T-Z system.''' 
           
        if self.loc_src[0] == loc_src[0] and self.loc_src[1] == loc_src[1]:
            return
        else:            
            self.loc_src = loc_src
            self.back_azimuth_array = np.zeros(self.n_station)
            self.distance_array = np.zeros(self.n_station)
            for i in range(self.n_station):
                dist, alpha12, alpha21 = gps2dist_azimuth(self.loc_src[0], self.loc_src[1],
                                                          self.loc_stations[i][0],
                                                          self.loc_stations[i][1])

                self.back_azimuth_array[i] = np.deg2rad(alpha21)
                # distance in km.
                self.distance_array[i] = np.sqrt(np.add(np.power(dist/1000, 2),
                                                         np.power(self.loc_src[2]-self.loc_stations[i][2], 2)))
      
                                                     
    def update_sgt(self, sgts_list):
        '''Update sgt parameters once the SGT is reset.'''
        self.n_syn_lens_pnl = []
        self.n_syn_lens_srf = []
        # Reset SGT containers
        self.sgt_pnl = []
        self.sgt_srf = []
        for i in range(self.n_station):
            self.sgt_pnl.append(sgts_list[i][0])
            self.sgt_srf.append(sgts_list[i][1])
            # SGT shape: length, 3 , 6
            n_pnl_syn, _, _ = np.shape(sgts_list[i][0])
            n_srf_syn, _, _ = np.shape(sgts_list[i][1])
            self.n_syn_lens_pnl.append(n_pnl_syn)
            self.n_syn_lens_srf.append(n_srf_syn)

    def update_data(self, data_list):
        '''Update data. '''
        self.n_data_lens_pnl = []
        self.n_data_lens_srf = []
        # Reset data containers
        self.data_pnl_rtz = []
        self.data_srf_rtz = []
        for i in range(self.n_station):
            self.data_pnl_rtz.append(DENZ2RTZ(data_list[i][0], self.back_azimuth_array[i]))
            self.data_srf_rtz.append(DENZ2RTZ(data_list[i][1], self.back_azimuth_array[i]))

            # data shape: 3, length
            _, _pnl_length = np.shape(data_list[i][0])
            _, _srf_length = np.shape(data_list[i][1])
            self.n_data_lens_pnl.append(_pnl_length)
            self.n_data_lens_srf.append(_srf_length)   

   
    def cut_and_paste(self, mt):
        '''
        Caculate the waveform misfit for the given moment tensor
        by Cut-and-Paste.
        '''
        misfit_matrix = np.ones((self.n_station, self.n_phase))
        shift_matrix = np.zeros((self.n_station, self.n_phase)).astype(int)
        cc_matrix = np.ones((self.n_station, self.n_phase))

        # synthetic wavefrom
        for i in range(self.n_station):
           
            # synthetic trace in R-T-Z.
            pnl_syn_rtz = DSynRotateRTZ(mt, self.sgt_pnl[i],
                                        self.back_azimuth_array[i], self.n_syn_lens_pnl[i])
            srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                      self.back_azimuth_array[i], self.n_syn_lens_srf[i])
            
            k = 0
            # pnl
            for j in range(self.n_component):
                if j == 1:
                    continue
                else:                  
                    # paste: shift and misfit calculation.
                    _n_shift, _misfit, _cc = DPaste(self.data_pnl_rtz[i][j],
                                             self.n_tp_data[i],
                                             self.n_data_lens_pnl[i],
                                             pnl_syn_rtz[j],
                                             self.n_tp_sgt[i],
                                             self.n_syn_lens_pnl[i],
                                             self.n_tp_max_shift)
                    # Apply Pnl weighting
                    if not self.w_pnl:
                        misfit_matrix[i, k] = 0
                    else:
                        misfit_matrix[i, k] = _misfit * self.w_pnl
                    shift_matrix[i, k] = _n_shift
                    cc_matrix[i, k] = _cc

                    k += 1
            # s/surface
            for j in range(self.n_component):
                
                # paste: shift and misfit calculation.                                   
                _n_shift, _misfit, _cc = DPaste(self.data_srf_rtz[i][j],
                                         self.n_ts_data[i],
                                         self.n_data_lens_srf[i],
                                         srf_syn_rtz[j],
                                         self.n_ts_sgt[i],
                                         self.n_syn_lens_srf[i],
                                         self.n_ts_max_shift)                    

                if not self.w_srf:
                    misfit_matrix[i, k] = 0
                else:
                    misfit_matrix[i, k] = _misfit * self.w_srf

                shift_matrix[i, k] = _n_shift
                cc_matrix[i, k] = _cc
                k += 1

        return misfit_matrix, shift_matrix, cc_matrix


    def cut_and_paste_grad(self, mt):
        '''
        for body wave: ZR
            for Hije: -(ur - dr) * sin(baz)
            for Hijn: -(ur - dr) * cos(baz)
            for Hijz: (uz - dz)
        for surface wave: ZRT
            for Hije: -(ur - dr) * sin(baz) - (ut - dt) * cos(baz)
            for Hijn: -(ur - dr) * cos(baz) + (ut - dt) * sin(baz)
            for Hijz: (uz - dz)
        '''
        n_lens = 200
        H_eff_array = np.zeros((self.n_station, 2, 3, n_lens))       
        # synthetic wavefrom 
        for i in range(self.n_station):
            #print('==========================',i)
            #print(self.station_names[i])
            misfit_pnl_rz = np.zeros((2, n_lens))
            misfit_srf_rtz = np.zeros((3, n_lens))
            
            # synthetic trace in R-T-Z.
            pnl_syn_rtz = DSynRotateRTZ(mt, self.sgt_pnl[i],
                                            self.back_azimuth_array[i], self.n_syn_lens_pnl[i])
            srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                        self.back_azimuth_array[i], self.n_syn_lens_srf[i])

            # p/body wave
            k = 0
            for j in range(self.n_component):  
                if j == 1:
                    continue
                else:                             
                    # paste: misfit gradient calculation.
                    _, misfit_pnl, _ = DPaste(self.data_pnl_rtz[i][j],
                                             self.n_tp_data[i],
                                             self.n_data_lens_pnl[i],
                                             pnl_syn_rtz[j],
                                             self.n_tp_sgt[i],
                                             self.n_syn_lens_pnl[i],
                                             self.n_tp_max_shift, type='GRAD')   
                    if not self.w_pnl:
                        misfit_pnl = 0
                    else:
                        misfit_pnl = misfit_pnl * self.w_pnl

                    misfit_pnl_rz[k] = np.hstack([misfit_pnl, np.zeros(n_lens-len(misfit_pnl))])   
                    k += 1

            H_eff_array[i, 0, 0] = -np.sin(self.back_azimuth_array[i]) * misfit_pnl_rz[0]                                     
            H_eff_array[i, 0, 1] = -np.cos(self.back_azimuth_array[i]) * misfit_pnl_rz[0]
            H_eff_array[i, 0, 2] = misfit_pnl_rz[1]

                    
            # s/surface
            for j in range(self.n_component):
                # paste: misfit gradient calculation.                            
                _, misfit_srf, _ = DPaste(self.data_srf_rtz[i][j],
                                         self.n_ts_data[i],
                                         self.n_data_lens_srf[i],
                                         srf_syn_rtz[j],
                                         self.n_ts_sgt[i],
                                         self.n_syn_lens_srf[i],
                                         self.n_ts_max_shift, type='GRAD')
                if not self.w_srf:
                    misfit_srf = np.zeros_like(misfit_srf)
                else:
                    misfit_srf = misfit_srf * self.w_srf

                misfit_srf_rtz[j] = np.hstack([misfit_srf, np.zeros(n_lens-len(misfit_srf))]) 

            H_eff_array[i, 1, 0] =  -np.sin(self.back_azimuth_array[i]) * misfit_srf_rtz[0] - np.cos(self.back_azimuth_array[i]) * misfit_srf_rtz[1]
            H_eff_array[i, 1, 1] =  -np.cos(self.back_azimuth_array[i]) * misfit_srf_rtz[0] +  np.sin(self.back_azimuth_array[i]) * misfit_srf_rtz[1]
            H_eff_array[i, 1, 2] =  misfit_srf_rtz[2] 
            
        return H_eff_array

    def cut_and_paste_grad_mt(self, mt):
        '''
        for body wave: ZR
            for Hije: -(ur - dr) * sin(baz)
            for Hijn: -(ur - dr) * cos(baz)
            for Hijz: (uz - dz)
        for surface wave: ZRT
            for Hije: -(ur - dr) * sin(baz) - (ut - dt) * cos(baz)
            for Hijn: -(ur - dr) * cos(baz) + (ut - dt) * sin(baz)
            for Hijz: (uz - dz)
        '''
        n_lens = 200
        H_eff_array = np.zeros((self.n_station, 2, 3, n_lens))       
        # synthetic wavefrom 
        for i in range(self.n_station):
            #print('==========================',i)
            #print(self.station_names[i])
            misfit_pnl_rz = np.zeros((2, n_lens))
            misfit_srf_rtz = np.zeros((3, n_lens))
            
            # synthetic trace in R-T-Z.
            pnl_syn_rtz = DSynRotateRTZ(mt, self.sgt_pnl[i],
                                            self.back_azimuth_array[i], self.n_syn_lens_pnl[i])
            srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                        self.back_azimuth_array[i], self.n_syn_lens_srf[i]) 
            #pnl_syn_rtz = pnl_syn_all[i]
            #srf_syn_rtz = srf_syn_all[i]
            #print(len(self.data_pnl_rtz[i][0]), len(pnl_syn_rtz[0]), self.station_names[i])
            # p/body wave
            k = 0
            for j in range(self.n_component):  
                if j == 1:
                    continue
                else:                             
                    # paste: misfit gradient calculation.
                    _, misfit_pnl, _ = DPaste(self.data_pnl_rtz[i][j],
                                             0,
                                             self.n_data_lens_pnl[i],
                                             pnl_syn_rtz[j],
                                             0,
                                             self.n_syn_lens_pnl[i],
                                             self.n_tp_max_shift, type='GRAD')   
                    if not self.w_pnl:
                        misfit_pnl = [0]
                    else:
                        misfit_pnl = misfit_pnl * self.w_pnl

                    misfit_pnl_rz[k] = np.hstack([misfit_pnl, np.zeros(n_lens-len(misfit_pnl))])   
                    k += 1

            H_eff_array[i, 0, 0] = -np.sin(self.back_azimuth_array[i]) * misfit_pnl_rz[0]                                     
            H_eff_array[i, 0, 1] = -np.cos(self.back_azimuth_array[i]) * misfit_pnl_rz[0]
            H_eff_array[i, 0, 2] = misfit_pnl_rz[1]

                    
            # s/surface
            for j in range(self.n_component):
                # paste: misfit gradient calculation.                            
                _, misfit_srf, _ = DPaste(self.data_srf_rtz[i][j],
                                         0,
                                         self.n_data_lens_srf[i],
                                         srf_syn_rtz[j],
                                         0,
                                         self.n_syn_lens_srf[i],
                                         self.n_ts_max_shift, type='GRAD')
                if not self.w_srf:
                    misfit_srf = [0]
                else:
                    misfit_srf = misfit_srf * self.w_srf

                misfit_srf_rtz[j] = np.hstack([misfit_srf, np.zeros(n_lens-len(misfit_srf))]) 

            H_eff_array[i, 1, 0] =  -np.sin(self.back_azimuth_array[i]) * misfit_srf_rtz[0] - np.cos(self.back_azimuth_array[i]) * misfit_srf_rtz[1]
            H_eff_array[i, 1, 1] =  -np.cos(self.back_azimuth_array[i]) * misfit_srf_rtz[0] +  np.sin(self.back_azimuth_array[i]) * misfit_srf_rtz[1]
            H_eff_array[i, 1, 2] =  misfit_srf_rtz[2] 
            
        return H_eff_array
        

    def cut_and_paste_grad_dep(self, mt):
        '''
        for body wave: ZR
            for dHije_dz: -(ur - dr) * sin(baz) * m_ji
            for dHijn_dz: -(ur - dr) * cos(baz) * m_ji
            for dHijz_dz: (uz - dz) * m_ji
        for surface wave: ZRT
            for dHije_dz: [ -(ur - dr) * sin(baz) - (ut - dt) * cos(baz) ] * m_ji
            for dHijn_dz: [ -(ur - dr) * cos(baz) + (ut - dt) * sin(baz) ] * m_ji
            for dHijz_dz: (uz - dz) * m_ji
        '''
        n_lens = 200
        dH_eff_array = np.zeros((self.n_station, 2, 3, 6, n_lens))       
        # synthetic wavefrom 
        for i in range(self.n_station):
            misfit_pnl_rz = np.zeros((2, n_lens))
            misfit_srf_rtz = np.zeros((3, n_lens))
            
            # synthetic trace in R-T-Z.
            pnl_syn_rtz = DSynRotateRTZ(mt, self.sgt_pnl[i],
                                            self.back_azimuth_array[i], self.n_syn_lens_pnl[i])
            srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                        self.back_azimuth_array[i], self.n_syn_lens_srf[i])
            #pnl_syn_rtz = pnl_syn_all[i]
            #srf_syn_rtz = srf_syn_all[i]

            # p/body wave
            k = 0
            for j in range(self.n_component):  
                if j == 1:
                    continue
                else:                             
                    # paste: misfit gradient calculation.
                    _, misfit_pnl, _ = DPaste(self.data_pnl_rtz[i][j],
                                             0,
                                             self.n_data_lens_pnl[i],
                                             pnl_syn_rtz[j],
                                             0,
                                             self.n_syn_lens_pnl[i],
                                             self.n_tp_max_shift, type='GRAD')   
                    if not self.w_pnl:
                        misfit_pnl = [0]
                    else:
                        misfit_pnl = misfit_pnl * self.w_pnl

                    misfit_pnl_rz[k] = np.hstack([misfit_pnl, np.zeros(n_lens-len(misfit_pnl))])   
                    k += 1

            for k in range(len(mt)):
                dH_eff_array[i, 0, 0, k] = -np.sin(self.back_azimuth_array[i]) * misfit_pnl_rz[0] * mt[k]                                    
                dH_eff_array[i, 0, 1, k] = -np.cos(self.back_azimuth_array[i]) * misfit_pnl_rz[0] * mt[k]
                dH_eff_array[i, 0, 2, k] = misfit_pnl_rz[1] * mt[k]

                    
            # s/surface
            for j in range(self.n_component):
                # paste: misfit gradient calculation.                            
                _, misfit_srf, _ = DPaste(self.data_srf_rtz[i][j],
                                         0,
                                         self.n_data_lens_srf[i],
                                         srf_syn_rtz[j],
                                         0,
                                         self.n_syn_lens_srf[i],
                                         self.n_ts_max_shift, type='GRAD')
                if not self.w_srf:
                    misfit_srf = [0]
                else:
                    misfit_srf = misfit_srf * self.w_srf

                misfit_srf_rtz[j] = np.hstack([misfit_srf, np.zeros(n_lens-len(misfit_srf))]) 

            for k in range(len(mt)):
                dH_eff_array[i, 1, 0, k] =  (-np.sin(self.back_azimuth_array[i]) * misfit_srf_rtz[0] - np.cos(self.back_azimuth_array[i]) * misfit_srf_rtz[1]) * mt[k]
                dH_eff_array[i, 1, 1, k] =  (-np.cos(self.back_azimuth_array[i]) * misfit_srf_rtz[0] +  np.sin(self.back_azimuth_array[i]) * misfit_srf_rtz[1]) * mt[k]
                dH_eff_array[i, 1, 2, k] =  misfit_srf_rtz[2] * mt[k]
            
        return dH_eff_array

    def get_data_list_vector(self):
        """
        Returns the observed data as a flattened 1D vector, matching the
        structure of syn_shifted_all for UKF usage.

        Returns:
            y_obs_vector : np.ndarray, shape = (sum of all station component lengths,)
        """
        data_all = []
        for i in range(self.n_station):
            # --- Pnl: R,Z components (skip T) ---
            pnl_data = [self.data_pnl_rtz[i][j] for j in range(self.n_component) if j != 1]
            # --- Srf: R,T,Z components ---
            srf_data = [self.data_srf_rtz[i][j] for j in range(self.n_component)]
            # Flatten all components for current station
            station_vector = np.hstack(pnl_data + srf_data)
            data_all.append(station_vector)
        
        # Concatenate all stations into final 1D vector
        y_obs_vector = np.hstack(data_all)
        return y_obs_vector



    def compute_syn_shifted_vector(self, mt):
        """
        Computes shifted synthetics for a given moment tensor (mt) and returns
        [pnl_syn(R,Z), srf_syn(R,T,Z)] for each station as a flattened 1D vector
        for UKF usage.
        """
        syn_all = []
        for i in range(self.n_station):
            # Synthesize Pnl & Srf three-component waveforms
            
            pnl_syn_rtz = DSynRotateRTZ(mt, self.sgt_pnl[i],
                                        self.back_azimuth_array[i], self.n_syn_lens_pnl[i])
            srf_syn_rtz = DSynRotateRTZ(mt, self.sgt_srf[i],
                                        self.back_azimuth_array[i], self.n_syn_lens_srf[i])
            
            # Pnl section (R,Z)
        
            pnl_shifted = []
            for j in range(self.n_component):
                if j == 1:
                    continue
                syn_shifted, _, _ = DPaste_shift_syn(
                    self.data_pnl_rtz[i][j],
                    self.n_data_lens_pnl[i],
                    pnl_syn_rtz[j],
                    self.n_tp_max_shift
                )
                pnl_shifted.append(syn_shifted)
     
            # Srf section (R,T,Z)
            srf_shifted = []
            for j in range(self.n_component):
                syn_shifted, _, _ = DPaste_shift_syn(
                    self.data_srf_rtz[i][j],
                    self.n_data_lens_srf[i],
                    srf_syn_rtz[j],
                    self.n_ts_max_shift
                )
                srf_shifted.append(syn_shifted)

            # Flatten pnl + srf for current station
            station_vector = np.hstack(pnl_shifted + srf_shifted)
            syn_all.append(station_vector)
        
        # Concatenate all stations into 1D vector
        y_syn_vector = np.hstack(syn_all)
        return y_syn_vector
