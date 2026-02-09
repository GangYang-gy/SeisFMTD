# -----------------------------------------------------------------------------
"""
Uncertainty estimation via the Hamiltonian Monte Carlo (HMC) method
for seismic moment tensor and source depth inversion.

This module provides tools for seismic moment tensor inversion and source
location estimation using Hamiltonian Monte Carlo (HMC) with optional
Dual-Averaging step-size adaptation.

Original Author: Liang Ding (myliang.ding@mail.utoronto.ca)
Modified By:     Gang Yang (gangy.yang@mail.utoronto.ca)
"""
# -----------------------------------------------------------------------------

import os
import pickle
import numpy as np
from scipy.linalg import cholesky
from pyrocko import moment_tensor as pmt
from MTfit.convert.moment_tensor_conversion import MT6_Tape
from pyCAPLunar.DCAPUtils import mag2moment
from pyCAPLunar.DCAP import DCAP
from pyCAPLunar.DPrepare import _prepare_N_station_sgt, _prepare_N_station_data
from pyCAPLunar.GTools import ENU2NED, NEZ2ENZ
from seisgen.util_SPECFEM3D import get_proc_name
from seisgen.util_SPECFEM3D.ibool_reader import DEnquire_Element
from seisgen.greens_function.sgt_reader import DEnquire_SGT
from seisgen.math.interp_grad import grad_sgt
import utm
from MTTools.DMomentTensors import DMT_enz


def convert_to_tapeq(mt_enu):
    """
    Converts an ENU moment tensor to Tape parameters q.

    Transforms the moment tensor from East-North-Up (ENU) convention into 
    source parameters: strike, dip, slip, magnitude, delta, and gamma.

    Args:
        mt_enu (np.ndarray): Moment tensor in ENU convention.

    Returns:
        np.ndarray: Array containing [strike, dip, slip, magnitude, delta, gamma].
    """
    mt_ned = ENU2NED(mt_enu)
    MT6 = np.array([
        mt_ned[0], mt_ned[1], mt_ned[2],
        np.sqrt(2)*mt_ned[3],
        np.sqrt(2)*mt_ned[4],
        np.sqrt(2)*mt_ned[5]
    ])
    gamma, delta, strike, cosdip, slip = MT6_Tape(MT6)
    m = pmt.values_to_matrix(mt_ned)
    moment = pmt.MomentTensor(m=m).scalar_moment()
    mag = (np.log10(moment) - 9.1)/1.5
    q = np.array([
        np.rad2deg(strike)[0],
        np.rad2deg(np.arccos(cosdip))[0],
        np.rad2deg(slip)[0],
        mag,
        np.rad2deg(delta)[0],
        np.rad2deg(gamma)[0]
    ])
    return q


# moment tensor and Depth.
class DHMC_MTD(DCAP):
    """
    Hamiltonian Monte Carlo solver for moment tensor and source location.

    Inherits from DCAP to leverage waveform processing and synthetic generation.
    Supports HMC sampling with optional Dual-Averaging step-size adaptation.
    """

    def __init__(self, sgtMgr,sgt_database_dir, mag, 
                 station_names, df, df_data,
                 p_freqmin, p_freqmax,
                 s_freqmin, s_freqmax,
                 n_p_length,n_s_length,
                 data_n_stations, loc_src, loc_stations,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 massinv=None,
                 n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=5.0,
                 reject_rate=1.0,
                 taper_scale=-0.4, taper_rate_srf=0.0,
                 cc_threshold=0.2,
                 amplitude_ratio_threshold=np.inf,   #2.0
                 filter = False,
                 job_id=None):

        # Store input parameters
        self.sgtMgr = sgtMgr
        self.sgt_database_dir = sgt_database_dir
        self.station_names = station_names
        self.loc_stations = loc_stations
        self.df = df
        self.df_data = df_data
        self.p_freqmin = p_freqmin
        self.p_freqmax = p_freqmax
        self.s_freqmin = s_freqmin
        self.s_freqmax = s_freqmax
        self.n_tp_data = n_tp_data
        self.n_ts_data = n_ts_data
        self.n_p_length = n_p_length
        self.n_s_length = n_s_length 
        self.data_n_stations = data_n_stations 
        self.lat_src = loc_src[0] 
        self.lon_src = loc_src[1] 
        self.filter=filter

        self.mag = mag
        self.NSPEC = 14720   

        sgts_list = self.get_sgt_list_n_stations(loc_src)
        data_list = self.get_data_list_n_stations(data_n_stations)
        
        super().__init__(sgts_list, data_list,
                         loc_src, loc_stations,station_names,
                         self.n_tp_sgt, self.n_ts_sgt,
                         n_tp_data, n_ts_data,
                         n_tp_max_shift, n_ts_max_shift,
                         df, n_phase, n_component,
                         w_pnl, w_srf,
                         misfit_threshold,
                         reject_rate,
                         taper_scale, taper_rate_srf,
                         cc_threshold,
                         amplitude_ratio_threshold)

        # HMC parameters
        self.sgts_list = sgts_list
        self.data_list = data_list
        self.sigma_d = 0.5#0.3
        self.mag = mag
        self.n_params = 7
        self.b_initial_q0 = False

        if massinv is None:
            massinv = np.eye(self.n_params)
        else:
            self.massinv = massinv

        # Sampling storage
        self.b_save_samples = True
        self.saving_dir = None
        self.job_id = job_id

        # Initialize Dual Averaging parameters with default values
        self.set_hmc_da_params()

    def get_sgt_list_n_stations(self, loc_src):
        """
        Prepare SGTs interpolated to the source location for all stations.

        Args:
            loc_src (list): Source location [lat, lon, depth].

        Returns:
            list: Interpolated SGTs per station.
        """

        self.find_point(loc_src)

        interp_sgt_n_stations = []

        for i in range(len(self.station_names)):
            interp_sgt = self.sgtMgr.get_sgt_HMC(self.station_names[i])
            interp_sgt_n_stations.append(interp_sgt)
     
        self.n_tp_sgt = self.n_tp_data
        self.n_ts_sgt = self.n_ts_data
        
        sgt_list_n_stations = _prepare_N_station_sgt(interp_sgt_n_stations, self.df,
                                                     self.p_freqmin, self.p_freqmax,
                                                     self.s_freqmin, self.s_freqmax,
                                                     self.n_tp_sgt, self.n_p_length,
                                                     self.n_ts_sgt, self.n_s_length, filter=self.filter)

        return sgt_list_n_stations


    def get_data_list_n_stations(self, data_n_stations):
        """
        Prepare observed data for all stations.

        Args:
            data_n_stations (list): Observed data arrays.

        Returns:
            list: Prepared data per station.
        """
        data_list_n_stations = _prepare_N_station_data(data_n_stations, 
                                                    self.df_data, self.df,
                                                    self.p_freqmin, self.p_freqmax,
                                                    self.s_freqmin, self.s_freqmax,
                                                    self.n_tp_data, self.n_p_length,
                                                    self.n_ts_data, self.n_s_length,filter=self.filter)  
        return data_list_n_stations

    # ---------------- Mechanics ----------------
    def create_p(self):
        """
        Generates momentum vector for HMC sampling.

        Returns:
            np.ndarray: Momentum vector of length n_params.
        """
        return np.random.normal(loc=0, scale=1.0, size=self.n_params)

    def Uq(self, q, b_sgt=True):
        """
        Computes the potential energy (negative log-posterior).

        Args:
            q (np.ndarray): Parameter vector.
            b_sgt (bool): Whether to update SGTs for depth.

        Returns:
            float: Potential energy value.
        """
        if b_sgt:
            update_loc_src = [self.lat_src,self.lon_src,q[-1]]
            sgt_list = self.get_sgt_list_n_stations(update_loc_src)
            self.update_sgt(sgt_list)                             

        mt0 = q[:-1].copy()*self.scaling
        mt0[3:] = 2 * mt0[3:] 
       
        misfit_matrix, _, _ = self.cut_and_paste(mt0)
        potential = np.sum(misfit_matrix) / (self.n_station) / (2.0 * np.power(self.sigma_d, 2))
        
        return potential


    def Kp(self, p):
        """
        Computes the kinetic energy.

        Args:
            p (np.ndarray): Momentum vector.

        Returns:
            float: Kinetic energy.
        """
        return 0.5 * p.T @ (self.massinv @ p)


    def dUdq(self, q):
        """
        Computes gradient of the potential energy w.r.t. q.

        Updates self.dU_dqi with gradients for moment tensor components and depth.
        """

        self.dU_dqi=np.zeros(self.n_params)

        # loc_src=[lat, long, depth in km] !!!depth is positive
        source = [self.lat_src,self.lon_src,q[-1]]
        sgts_list = self.get_sgt_list_n_stations(source)

        # Gradient w.r.t. moment tensor
        grad = np.zeros(6)
        
        mt0 = q[:-1].copy()*self.scaling
        mt0[3:] = 2 * mt0[3:] 
        H_eff_array = self.cut_and_paste_grad_mt(mt0) 
        grad_matrix = np.zeros((6, self.n_station, 3*2))

        n_lens = 200

        for i in range(self.n_station): 
            
            SGT_pnl_nez = sgts_list[i][0]      
            SGT_srf_nez = sgts_list[i][1]
            SGT_pnl_enz = NEZ2ENZ(SGT_pnl_nez)
            SGT_srf_enz = NEZ2ENZ(SGT_srf_nez)
            k = 0         
            for j in range(self.n_component):  
                for l in range(6):
                    grad_matrix[l, i, k] = np.sum(np.hstack([SGT_pnl_enz[:, j, l], np.zeros(n_lens-len(SGT_pnl_enz[:, j, l]))]) * H_eff_array[i, 0, j]) 
                    if l >=3 :
                        grad_matrix[l, i, k] = grad_matrix[l, i, k] * 2
                k += 1             
               
            for j in range(self.n_component):      
                for l in range(6):    
                    grad_matrix[l, i, k] = np.sum(np.hstack([SGT_srf_enz[:, j, l], np.zeros(n_lens-len(SGT_srf_enz[:, j, l]))]) * H_eff_array[i, 1, j]) 
                    if l >=3 :
                        grad_matrix[l, i, k] = grad_matrix[l, i, k] * 2
                k += 1      

        for l in range(6):     #Mxx,Myy,Mzz,Mxy,Mxz,Myz in ENU convention
            grad[l] = np.sum(grad_matrix[l, :, :]) / (self.n_station) / np.power(self.sigma_d, 2)  

        self.dU_dqi[:-1] = grad * self.scaling

        # Gradient w.r.t. depth
        dH_eff_array = self.cut_and_paste_grad_dep(mt0)   # (n_station, 2, 3, n_lens, 6)
        grad_matrix = np.zeros((self.n_station, 3*2))

        self.sgtMgr.idx_element = self.sgtMgr.element_index - 1
        self.sgtMgr.proc_name = get_proc_name(self.sgtMgr.idx_processor)

        ibool_file = os.path.join(str(self.sgtMgr.model3D_folder), str(self.sgtMgr.proc_name)+str("_ibool.bin"))
        self.sgtMgr.idx_glls = DEnquire_Element(ibool_file,  self.sgtMgr.idx_element, self.NSPEC)

        for i in range(self.n_station):
            
            dir_string = os.path.join(str(self.sgt_database_dir),
                                        str(self.station_names[i]),
                                        str(self.sgtMgr.proc_name))
            sgt_data_path = dir_string + str("_sgt_data.bin")
            sgt_hder_path = dir_string + str("_header.hdf5")
            sgt_arr_nez = DEnquire_SGT(sgt_data_path, sgt_hder_path, self.sgtMgr.idx_glls)   

            _, _, dsgt_dz_nez = grad_sgt(self.sgtMgr.idx_element, self.NSPEC, self.sgtMgr.model3D_folder, \
                                                self.sgtMgr.idx_processor, self.sgtMgr.xi, self.sgtMgr.eta, self.sgtMgr.gamma, sgt_arr_nez) #(n_len, 3, 6)
 
            dsgt_dz = NEZ2ENZ(dsgt_dz_nez)
            k = 0   
            #print(np.shape(dsgt_dz), np.shape(dH_eff_array))  #(310, 3, 6) (10, 2, 3, 6, 200)
            for j in range(self.n_component):  
                for l in range(6):
                    dsgt_dz_pnl = np.hstack([dsgt_dz[:self.n_ts_data[i], j, l], np.zeros((n_lens-len(dsgt_dz[:self.n_ts_data[i], j, l])))])
                    grad_matrix[i, k] += np.dot(dsgt_dz_pnl, dH_eff_array[i, 0, j, l,:])
        
                k += 1             
               
            for j in range(self.n_component):      
                for l in range(6):
                    dsgt_dz_srf = np.hstack([dsgt_dz[self.n_ts_data[i]:self.n_ts_data[i] + self.n_s_length, j, l], np.zeros(n_lens-len(dsgt_dz[self.n_ts_data[i]:self.n_ts_data[i] + self.n_s_length, j, l]))])
                    grad_matrix[i, k] += np.dot(dsgt_dz_srf, dH_eff_array[i, 1, j, l, :])
                
                k += 1     
        
        self.dU_dqi[-1] = np.sum(grad_matrix[:, :]) / (self.n_station) / np.power(self.sigma_d, 2) * 1000    #grad is for meter, while FD is for km  
        
        
    def dUdq_FD(self, q):
        """
        Computes the gradient of potential energy using finite differences.

        Args:
            q (np.ndarray): Parameter vector.
            epsilon (float): Small perturbation for finite difference.

        Returns:
            np.ndarray: Gradient vector of length n_params.
        """
        self.dU_dqi=np.zeros(self.n_params)
        
        for i in range(self.n_params):
        
            if i< 6:
                q1 = q.copy()
                q1[i] = q[i] * (1 + 0.01)
                u1 = self.Uq(q1,b_sgt = False)
                q1[i] = q[i] * (1 - 0.01)
                u2 = self.Uq(q1,b_sgt = False)
                self.dU_dqi[i] = (u1 - u2) / (0.02 * q[i])
            
            else:
                b_sgt = True
                q1 = q.copy()
                q1[i] = q[i] + 0.1 
                u1 = self.Uq(q1,b_sgt)
                q1[i] = q[i] - 0.1
                u2 = self.Uq(q1,b_sgt) 
                self.dU_dqi[i] = (u1 - u2) / 0.2


    def find_point(self, loc_src):
        """
        Locate the source point within the SGT (Strain Green's Tensor) grid.

        Args:
            loc_src (list or np.ndarray): Source location [latitude, longitude, depth in km].
                                           Depth should be positive (km).

        Notes:
            - The SGT manager (`self.sgtMgr`) uses this function to determine the
              local element indices and natural coordinates (xi, eta, gamma) 
              corresponding to the source point.
            - Depth is converted to meters internally.
        """
        source = loc_src.copy()
        source[2] = 1000 * np.fabs(source[2])
        
        #utm_x, utm_y,_,_ = utm.from_latlon(source[0], source[1])

        #_, _, _,_, \
        #_, _, _, \
        #self.sgtMgr.idx_processor, self.sgtMgr.element_index, \
        #self.sgtMgr.xi, self.sgtMgr.eta, self.sgtMgr.gamma = self.sgtMgr.find(x=utm_x, y=utm_y, z=source[2], mode='UTM',b_depth=True)
        _, _, _,_, \
        _, _, _, \
        self.sgtMgr.idx_processor, self.sgtMgr.element_index, \
        self.sgtMgr.xi, self.sgtMgr.eta, self.sgtMgr.gamma = self.sgtMgr.find(x=source[0], y=source[1], z=source[2],b_depth=True)
        
            
    def set_q(self, q):
        """
        Sets the starting position for the sampler and initializes scaling.

        Args:
            q (np.ndarray): Initial parameter vector.
        """
        if len(q) < self.n_params:
            print(f"Bad q0 with {len(q)} parameters, while {self.n_params} are required.")
            return

        self.q0 = q.copy()
        if not self.b_initial_q0:
            print("set_q")
            self.scaling = mag2moment(self.mag)
            self.q0[:-1] = self.q0[:-1] / self.scaling
        
        self.b_initial_q0 = True
        self.dUdq(self.q0)
        #print(self.dU_dqi)
        #exit()
        
       
    def set_hmc_da_params(self, da_t0=10, da_gamma=0.05, da_kappa=0.75, 
                          target_acceptance=0.65, warmup_fraction=0.4, path_length=5):
        """
        Configures parameters for the Dual Averaging step-size adaptation.

        Args:
            da_t0 (float): Stabilization parameter for early iterations.
            da_gamma (float): Scale of the adaptation.
            da_kappa (float): Learning rate decay power.
            target_acceptance (float): The desired HMC acceptance ratio.
            warmup_fraction (float): Fraction of samples used to tune epsilon.
            path_length (float): Integration time for the leapfrog integrator.
        """
        self.da_t0 = da_t0
        self.da_gamma = da_gamma
        self.da_kappa = da_kappa
        self.target_acceptance = target_acceptance
        self.warmup_fraction = warmup_fraction
        self.path_length = path_length
        
    def _leapfrog(self, q, p, epsilon, L):
        """
        Performs Leapfrog integration to simulate Hamiltonian dynamics.

        Args:
            q (np.ndarray): Current position.
            p (np.ndarray): Current momentum.
            epsilon (float): Integration step size.
            L (int): Number of integration steps.

        Returns:
            tuple: (new_q, new_p) after integration.
        """
        q_new, p_new = q.copy(), p.copy()
       
        # Initial half-step for momentum
        self.dUdq(q_new)
        p_new -= 0.5 * epsilon * self.dU_dqi
        
        # Full steps for position and momentum
        for _ in range(L):
            q_new += epsilon * (self.massinv @ p_new)
            self.dUdq(q_new) 
            if _ != L - 1:
                p_new -= epsilon * self.dU_dqi

        # Final half-step for momentum
        p_new -= 0.5 * epsilon * self.dU_dqi

        return q_new, -p_new

    def _find_reasonable_epsilon(self, q):
        """
        Heuristic to find a suitable starting step size (epsilon).

        Args:
            q (np.ndarray): Current position.

        Returns:
            float: A reasonable initial epsilon.
        """
        print("Finding reasonable epsilon...")
        eps = 1.0
        p = self.create_p()

        H_current = self.Uq(q) + self.Kp(p)
        q_new, p_new = self._leapfrog(q.copy(), p.copy(), eps, 1)
        H_new = self.Uq(q_new) + self.Kp(p_new)

        dH = H_current - H_new
        alpha = np.exp(dH)
        alpha = max(alpha, 1e-12) 
        a = 1 if alpha > 0.5 else -1

        # Adjust epsilon by factors of 2 until the acceptance probability crosses 0.5
        while (alpha ** a) > (2.0 ** (-a)):
            eps *= 2.0 ** a
            q_new, p_new = self._leapfrog(q.copy(), p.copy(), eps, 1)
            H_new = self.Uq(q_new) + self.Kp(p_new)
            alpha = np.exp(H_current - H_new)
            alpha = max(alpha, 1e-12)
            
        print(f"Found reasonable epsilon = {eps:.4f}")
        return eps

    def hmc_da(self, n_sample, n_step=20):
        """
        Executes HMC with Dual-Averaging for automated step-size tuning.

        Args:
            n_sample (int): Target number of samples.
            n_step (int): Default number of leapfrog steps if path_length is None.

        Returns:
            tuple: (samples, mean, misfits)
        """
        samples = []
        misfits = []

        current_q = self.q0.copy()
        U_current = self.Uq(current_q)

        accepted = 0

        # ---------- warmup length ----------
        M_adapt = int(n_sample * self.warmup_fraction)
        total_iter = M_adapt + n_sample

        # ---------- initial epsilon ----------
        eps0 = self._find_reasonable_epsilon(current_q)

        mu = np.log(10.0 * eps0)
        H_bar = 0.0
        log_eps = np.log(eps0)
        eps_bar = eps0

        # ======================================
        # main chain
        # ======================================

        for m in range(1, total_iter + 1):

            # ----- step size -----
            epsilon = float(np.exp(log_eps)) if m <= M_adapt else float(eps_bar)
            #epsilon = max(epsilon, 1e-6)

            # ----- leapfrog steps from path length -----
            if self.path_length:
                Lm = max(1, int(round(self.path_length / max(epsilon, 1e-16))))
            else:
                Lm = n_step
            print("Lm, epsilon", Lm, epsilon) 

            # ----- proposal -----
            p0 = self.create_p()
            q_prop, p_prop = self._leapfrog(current_q.copy(), p0.copy(), epsilon, Lm)
            
            # ----- energy -----
            U_prop = self.Uq(q_prop)

            H_current = U_current + self.Kp(p0)
            H_prop = U_prop + self.Kp(p_prop)

            dH = H_current - H_prop

            # ----- acceptance probability -----
            alpha = min(1.0, np.exp(dH))
    
            if np.log(np.random.rand()) < min(0.0, dH):
                current_q = q_prop
                U_current = U_prop
                accepted += 1
                print(f"[Accepted] {accepted}/{m}")
                print("Accept q:", np.hstack([convert_to_tapeq(current_q[:-1].copy()*self.scaling), current_q[-1]]))
                    
            else:
                print(f"[Rejected] {m - accepted}/{m}")
                print("Reject q:", np.hstack([convert_to_tapeq(q_prop[:-1].copy()*self.scaling), q_prop[-1]]))

            # ----- record chain -----
            current_q_ = current_q.copy()
            current_q_[:-1] = current_q_[:-1] * self.scaling
            samples.append(current_q_)
            misfits.append(U_current)
    
            # ======================================
            # Dual Averaging update (warmup only)
            # ===================================== 
            if m <= M_adapt:
                eta = 1.0 / (m + self.da_t0)
                H_bar = (1.0 - eta) * H_bar + eta * (self.target_acceptance - alpha)
                log_eps = mu - (np.sqrt(m) / self.da_gamma) * H_bar
                power = m ** (-self.da_kappa)
                eps_bar = np.exp(power * log_eps + (1.0 - power) * np.log(eps_bar))
            else:
                log_eps = np.log(eps_bar)

            if len(samples) != 0  and len(samples) % 100 ==0:
                step = len(samples)
                self.save_to_file(f"Samples_N{n_sample}_{step}", np.asarray(samples), np.asarray(misfits))

        # ======================================
        # burn-in removal
        # ======================================
        samples = np.asarray(samples, dtype=float)
        misfits = np.asarray(misfits, dtype=float)

        if len(samples) <= M_adapt:
            raise RuntimeError("Warmup larger than chain length")

        samples_post = samples[M_adapt:]
        misfits_post = misfits[M_adapt:]

        mean = np.mean(samples_post, axis=0)

        acceptance_rate = accepted / total_iter

        print(f"Final acceptance rate = {acceptance_rate:.4f}")
        print(f"Adapted epsilon = {eps_bar:.6g}")

         # ----- save -----
        if self.b_save_samples:
            self.save_to_file(
                f"Samples_N{n_sample}_",
                samples_post,
                misfits_post
            )
            self.acceptance_rate = acceptance_rate
            self.epsilon = eps_bar
            self.M_adapt = M_adapt

        return samples_post, mean, misfits_post


    def set_saving_dir(self, saving_dir):
        """Sets the directory where sample files will be stored."""
        self.saving_dir = saving_dir

    def save_to_file(self, name_str, samples, misfit, format='.pkl'):
        """
        Serializes sample and misfit data to a file.

        Args:
            name_str (str): Base filename.
            samples (np.ndarray): Collected parameter samples.
            misfit (np.ndarray): Corresponding misfit values.
            format (str): File extension (default .pkl).
        """
        if self.job_id is None:
            file_path = str(self.saving_dir) + str(name_str) + str(format)
        else:
            file_path = str(self.saving_dir) + str(self.job_id) + str('_') + str(name_str) + str(format)
        
        with open(file_path, 'wb') as f:
            pickle.dump(samples, f)
            pickle.dump(misfit, f)


    def ukf(self, num_iterations=30, alpha=1e-3, beta=2, kappa=0, eps=1e-5):
        """
        Estimates the moment tensor using an Unscented Kalman Filter.

        Args:
            num_iterations (int): Maximum filter iterations.
            alpha (float): Determines the spread of sigma points.
            beta (float): Incorporates prior knowledge of the distribution.
            kappa (float): Secondary scaling parameter.
            eps (float): Convergence threshold for state change.

        Returns:
            tuple: (final_state, trajectory, covariance_matrix)
        """

        # --- init ---
        #self.process_noise_scale = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e7])
        #self.sigma_d = 0.001
        self.process_noise_scale = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e7])
        self.sigma_d = 0.001
        
        x = self.q0.copy()  # state vector
        L = self.n_params
        lam = alpha**2 * (L + kappa) - L

        # Initialize weights for mean and covariance
        Wm = np.full(2 * L + 1, 0.5 / (L + lam))
        Wm[0] = lam / (L + lam)
        Wc = Wm.copy()
        Wc[0] += 1 - alpha**2 + beta

        P = np.diag(self.process_noise_scale)
        y_obs = self.get_data_list_vector()
        R = np.eye(len(y_obs)) * self.sigma_d**2

        trajectory = [np.hstack([convert_to_tapeq(x.copy()[:-1] * self.scaling), x.copy()[-1]])]

        for i in range(num_iterations):
            print(f"🔄 UKF iteration {i+1} / {num_iterations}")

            # Generate Sigma Points
            try:
                sqrtP = cholesky((L + lam) * P)
            except np.linalg.LinAlgError:
                P += np.eye(L) * 1e-8
                sqrtP = cholesky((L + lam) * P)

            sigmas = np.hstack([
                x.reshape(-1, 1),
                x.reshape(-1, 1) + sqrtP,
                x.reshape(-1, 1) - sqrtP
            ]).T  # shape (2L+1, L)

            # Predict observation for each sigma point
            Y = []
            for s in sigmas:
                depth = float(s[-1])
                update_loc_src = [self.lat_src, self.lon_src, depth]
                sgt_list = self.get_sgt_list_n_stations(update_loc_src)
                self.update_sgt(sgt_list)

                mt0 = s[:-1] * self.scaling
                mt0[3:] *= 2
                y_s, _ = self.compute_syn_shifted_vector(mt0)
                Y.append(y_s)
            Y = np.asarray(Y)  # shape (2L+1, obs_dim)

            # --- mean and covariance ---
            y_mean = Wm @ Y
            Y_centered = (Y - y_mean).T  # (obs_dim, 2L+1)
            P_yy = (Y_centered * Wc) @ Y_centered.T + R
            P_yy += np.eye(P_yy.shape[0]) * 1e-8  # regularize

            X_centered = (sigmas - x).T  # (L, 2L+1)
            P_xy = X_centered @ (Wc[:, None] * Y_centered.T)

            # --- Kalman gain and update ---
            try:
                P_yy_inv = np.linalg.inv(P_yy)
            except np.linalg.LinAlgError:
                P_yy += np.eye(P_yy.shape[0]) * 1e-6
                P_yy_inv = np.linalg.inv(P_yy)

            K = P_xy @ P_yy_inv

            x_new = x + K @ (y_obs - y_mean)
            trajectory.append(np.hstack([convert_to_tapeq(x_new[:-1] * self.scaling), x_new[-1]]))

            # convergence check
            dx_norm = np.linalg.norm(x_new - x)
            if dx_norm < eps:
                print(f"✅ Converged at iter {i+1} (Δx={dx_norm:.2e} < {eps})")
                x = x_new
                break
            if dx_norm > 50:  
                print(f"Warning: State update too large at iteration {i+1}: {dx_norm}")
                break

            # covariance update
            P = P - K @ P_yy @ K.T
            P = 0.5 * (P + P.T)  # symmetrize
            eigs = np.linalg.eigvals(P)
            min_eig = np.min(eigs.real)
            if min_eig < 1e-6:
                P += np.eye(L) * (1e-6 - min_eig)

            x = x_new
            print("   → Updated state:", np.hstack([convert_to_tapeq(x[:-1] * self.scaling), x[-1]]))
        else:
            print(f"⚠️ UKF did not converge after {num_iterations} iterations.")

        return np.hstack([convert_to_tapeq(x[:-1] * self.scaling), x[-1]]), trajectory, P


    def compute_misfit(self, q=None, update_depth=False):
        """
        Compute misfit between observed and predicted travel times for a given point q.

        Args:
            q (np.ndarray): source location [x, y, depth, (optional t0)]
            save_csv_path (str): if given, will save detailed predictions and observations to a CSV file

        Returns:
            misfit (float): total squared misfit / (2 * sigma_d^2)
        """
        if q is None:
            mt0 = self.q0[:-1].copy() * self.scaling
            mt0[3:] = mt0[3:] * 2
        else:
            mt0 = mag2moment(q[3]) * DMT_enz(np.deg2rad(q[0]), np.deg2rad(q[1]),
                                             np.deg2rad(q[2]), np.deg2rad(q[4]),
                                             np.deg2rad(q[5])) 

        if update_depth:
            update_loc_src = [self.lat_src, self.lon_src, q[-1]]
            sgt_list = self.get_sgt_list_n_stations(update_loc_src)
            self.update_sgt(sgt_list)  
      
        sgt_shifted, misfit_all = self.compute_syn_shifted_vector(mt0)
 
        data_list = self.get_data_list_vector()

        residual = sgt_shifted - data_list
        misfit_l2 = 0.5 * np.sum(residual**2) / (2.0 * np.power(self.sigma_d, 2))

        return misfit_l2

