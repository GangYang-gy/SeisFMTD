# -----------------------------------------------------------------------------
"""
Uncertainty estimation via the Hamiltonian Monte Carlo (HMC) method.

This module provides tools for seismic moment tensor inversion and uncertainty 
quantification using Hamiltonian Monte Carlo (HMC) and Unscented Kalman Filter 
(UKF) approaches.

Original Author: Liang Ding (myliang.ding@mail.utoronto.ca)
Modified By:     Gang Yang (gangy.yang@mail.utoronto.ca)
"""
#-----------------------------------------------------------------------------

import pickle
import numpy as np
from scipy.linalg import cholesky
from pyrocko import moment_tensor as pmt
from MTfit.convert.moment_tensor_conversion import MT6_Tape

from pyCAPLunar.DCAPUtils import mag2moment
from pyCAPLunar.DCAP import DCAP
from pyCAPLunar.GTools import NEZ2ENZ, ENU2NED


def convert_to_tapeq(mt_enu):
    """
    Converts an ENU moment tensor to Tape parameters.

    Transforms the moment tensor from East-North-Up (ENU) convention into 
    source parameters: strike, dip, slip, magnitude, delta, and gamma.

    Args:
        mt_enu (np.ndarray): Moment tensor in ENU convention.

    Returns:
        np.ndarray: Array containing [strike, dip, slip, magnitude, delta, gamma].
    """
    mt_ned = ENU2NED(mt_enu)
    
    # Scale off-diagonal elements by sqrt(2) to comply with MT6 vector format
    mt6_vector = np.array([
        mt_ned[0], mt_ned[1], mt_ned[2],
        np.sqrt(2) * mt_ned[3],
        np.sqrt(2) * mt_ned[4],
        np.sqrt(2) * mt_ned[5]
    ])
    
    gamma, delta, strike, cos_dip, slip = MT6_Tape(mt6_vector)
    mt_matrix = pmt.values_to_matrix(mt_ned)
    moment = pmt.MomentTensor(m=mt_matrix).scalar_moment()
    
    # Standard formula for Moment Magnitude (Mw)
    magnitude = (np.log10(moment) - 9.1) / 1.5
    
    return np.array([
        np.rad2deg(strike)[0],
        np.rad2deg(np.arccos(cos_dip))[0],
        np.rad2deg(slip)[0],
        magnitude,
        np.rad2deg(delta)[0],
        np.rad2deg(gamma)[0]
    ])


class DHMC_MT(DCAP):
    """
    Hamiltonian Monte Carlo (HMC) solver for seismic moment tensor components.
    
    Inherits from DCAP to leverage waveform processing and synthetic generation.
    This class implements HMC with Dual-Averaging (DA) for automated step-size
    tuning and a UKF implementation for state estimation.
    """

    def __init__(self, sgts_list, data_list,
                 mag, massinv,
                 loc_src, loc_stations, station_names,
                 n_tp_sgt, n_ts_sgt,
                 n_tp_data, n_ts_data,
                 n_tp_max_shift, n_ts_max_shift,
                 dt, n_phase=5, n_component=3,
                 w_pnl=1.0, w_srf=1.0,
                 misfit_threshold=5.0,
                 reject_rate=1.0,
                 taper_scale=-0.4, taper_rate_srf=0.0,
                 cc_threshold=0.2,
                 amplitude_ratio_threshold=np.inf,
                 job_id=None):
                
        """Initializes the DHMC solver with seismic data and HMC parameters."""

        super().__init__(sgts_list, data_list,
                         loc_src, loc_stations, station_names,
                         n_tp_sgt, n_ts_sgt,
                         n_tp_data, n_ts_data,
                         n_tp_max_shift, n_ts_max_shift,
                         dt, n_phase, n_component,
                         w_pnl, w_srf,
                         misfit_threshold,
                         reject_rate,
                         taper_scale, taper_rate_srf,
                         cc_threshold,
                         amplitude_ratio_threshold)

        self.sgts_list = sgts_list
        self.data_list = data_list
        self.sigma_d = 0.1
        self.mag = mag
        self.mass_inv = massinv
        self.n_params = 6
        self.b_initial_q0 = False

        # Physical constraints for [Strike, Dip, Slip, Mw, Delta, Gamma]
        self.boundaries = np.array([
            [0, 360],
            [0, 90],
            [-90, 90],
            [3, 6],
            [0, 180],
            [-30, 30]
        ])

        self.b_save_samples = True
        self.saving_dir = None
        self.job_id = job_id

        # Initialize Dual Averaging parameters with default values
        self.set_hmc_da_params()

    def _check_boundary(self, q, index):
        """
        Validates if the parameter at the given index respects physical limits.

        Args:
            q (np.ndarray): Current parameter vector.
            index (int): Index of the parameter to check.

        Returns:
            tuple: (is_within_bounds, boundary_value_if_clamped).
        """
        lower, upper = self.boundaries[index]
        if q[index] < lower:
            return False, lower
        if q[index] > upper:
            return False, upper
        return True, None

    def create_p(self):
        """
        Generates a momentum vector from a standard normal distribution.

        Returns:
            np.ndarray: Momentum vector of length n_params.
        """
        return np.random.normal(0, 1.0, size=self.n_params)

    def Uq(self, q):
        """
        Calculates the Potential Energy, defined as the negative log-posterior.

        Args:
            q (np.ndarray): Parameter vector (normalized).

        Returns:
            float: Potential energy value based on waveform misfit.
        """
        mt_scaled = q.copy() * self.scaling
        # Adjusting 4th, 5th, and 6th elements for tensor vectorization
        mt_scaled[3:] *= 2 
        
        misfit_matrix, _, _ = self.cut_and_paste(mt_scaled)
        # Weight misfit by the number of stations and data variance
        potential = np.sum(misfit_matrix) / (self.n_station * 2.0 * (self.sigma_d**2))
        return potential

    def Kp(self, p):
        """
        Calculates the Kinetic Energy.

        Args:
            p (np.ndarray): Momentum vector.

        Returns:
            float: Kinetic energy value.
        """
        return 0.5 * p.T @ (self.mass_inv @ p)

    def dUdq(self, q):  
        """
        Computes the gradient of the Potential Energy with respect to q.

        Updates the internal attribute self.dU_dqi with the calculated gradient.

        Args:
            q (np.ndarray): Current parameter vector.
        """
        grad = np.zeros(self.n_params)
        mt0 = q.copy() * self.scaling
        mt0[3:] = mt0[3:] * 2   
        
        # Calculate effective H-array for gradient via Cut-and-Paste
        H_eff_array = self.cut_and_paste_grad(mt0)  
        grad_matrix = np.zeros((6, self.n_station, 3 * 2))
        n_lens = 200

        for i in range(self.n_station): 
            SGT_pnl_nez = self.sgts_list[i][0]
            SGT_srf_nez = self.sgts_list[i][1]
            SGT_pnl_enz = NEZ2ENZ(SGT_pnl_nez)
            SGT_srf_enz = NEZ2ENZ(SGT_srf_nez)
            
            k = 0         
            # Process Pnl waves
            for j in range(self.n_component):  
                for l in range(6):
                    conv_sum = np.sum(np.hstack([SGT_pnl_enz[:, j, l], 
                                                 np.zeros(n_lens - len(SGT_pnl_enz[:, j, l]))]) * H_eff_array[i, 0, j])
                    grad_matrix[l, i, k] = conv_sum
                    if l >= 3:
                        grad_matrix[l, i, k] *= 2
                k += 1             
            
            # Process Surface waves
            for j in range(self.n_component):      
                for l in range(6):    
                    conv_sum = np.sum(np.hstack([SGT_srf_enz[:, j, l], 
                                                 np.zeros(n_lens - len(SGT_srf_enz[:, j, l]))]) * H_eff_array[i, 1, j]) 
                    grad_matrix[l, i, k] = conv_sum
                    if l >= 3:
                        grad_matrix[l, i, k] *= 2
                k += 1      

        # Sum contributions over all stations and components
        for l in range(6):
            grad[l] = np.sum(grad_matrix[l, :, :]) / self.n_station / np.power(self.sigma_d, 2)
        
        self.dU_dqi = grad * self.scaling

    def set_q(self, q):
        """
        Sets the starting position for the sampler and initializes scaling.

        Args:
            q (np.ndarray): Initial parameter vector.
        """
        if len(q) < self.n_params:
            print(f"Bad q0 with {len(q)} parameters, while {self.Nq} are required.")
            return
            
        self.q0 = q.copy()
        if not self.b_initial_q0:
            self.scaling = mag2moment(self.mag)
            self.q0 = q.copy() / self.scaling

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
            q_new += epsilon * (self.mass_inv @ p_new)
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

    def hmc(self, n_sample, epsilon, n_step=20, M_adapt=200):
        """
        Executes a standard Hamiltonian Monte Carlo sampling routine.

        Args:
            n_sample (int): Number of samples to collect.
            epsilon (float): Leapfrog step size.
            n_step (int): Number of leapfrog steps per sample.
            M_adapt (int): Warmup period for burn-in.

        Returns:
            tuple: (samples, mean, misfits)
        
        Sampling strategy:
        - The chain length is fixed to warmup_steps + n_sample iterations.
        - Burn-in is iteration-based (not acceptance-count-based) to maintain
          correct MCMC statistical properties.
        - Only post-warmup samples are returned and used for posterior statistics.
        """

        samples = []
        misfits = []

        current_q = self.q0.copy()
        U_current = self.Uq(current_q)

        accepted = 0
        total_iter = M_adapt + n_sample

        for m in range(1, total_iter + 1):
         
            # ---- sample momentum ----
            p0 = self.create_p()

            # ---- simulate Hamiltonian dynamics ----
            q_prop, p_prop = self._leapfrog(
            current_q.copy(),
            p0.copy(),
            epsilon,
            n_step
        )

            # ---- compute energy ----
            U_prop = self.Uq(q_prop)

            H_current = U_current + self.Kp(p0)
            H_prop = U_prop + self.Kp(p_prop)
    
            dH = H_current - H_prop

            # ---- numerical safety ----
            accept = False
            if np.log(np.random.rand()) < min(0.0, dH):
                accept = True

            # ---- accept / reject ----
            if accept:
                print(f"[Accepted] {accepted}/{m}")
                current_q = q_prop
                U_current = U_prop
                accepted += 1
            else:
                print(f"[Rejected] {m - accepted}/{m}")

            # ---- record chain state (IMPORTANT: always record) ----
            samples.append(current_q.copy() * self.scaling)
            misfits.append(self.Uq(current_q))

            # ---- controlled logging ----
            if i % 50 == 0:
                print(f"Iter {m}/{total_iter} | acc_rate = {accepted/m:.3f}")
    
        acceptance_rate = accepted / total_iter
        print(f"Final acceptance rate is {acceptance_rate:.4f} | {accepted}/{total_iter}")

        # =============================
        # burn-in removal
        # =============================
        samples = np.asarray(samples, dtype=float)
        misfits = np.asarray(misfits, dtype=float)

        if len(samples) <= M_adapt:
            raise RuntimeError("Burn-in larger than chain length")

        samples_post = samples[M_adapt:]
        misfits_post = misfits[M_adapt:]

        mean = np.mean(samples_post, axis=0)
        cov = np.cov(samples_post.T)

        acceptance_rate = accepted / total_iter
        print(f"Final acceptance rate = {acceptance_rate:.4f}")

        if self.b_save_samples:
            self.save_to_file(
                f"Samples_N{n_sample}_",
                samples_post,
                misfits_post
            )

        return samples_post, mean, misfits_post

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
            epsilon = max(epsilon, 1e-6)

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

            # ----- accept / reject -----
            if np.log(np.random.rand()) < min(0.0, dH):
                current_q = q_prop
                U_current = U_prop
                accepted += 1
                print(f"[Accepted] {accepted}/{m}")
                print("Accept q:", convert_to_tapeq(current_q.copy()*self.scaling))
            else:
                print(f"[Rejected] {m - accepted}/{m}")
                print("Reject q:", convert_to_tapeq(q_prop.copy()*self.scaling))
    

            # ----- record chain -----
            samples.append(current_q.copy() * self.scaling)
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

            # ----- controlled logging -----
            if m % 50 == 0:
                print(
                    f"Iter {m}/{total_iter} | "
                    f"eps={epsilon:.4g} | "
                    f"L={Lm} | "
                    f"acc_rate={accepted/m:.3f}"
                )
        
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
            file_path = f"{self.saving_dir}{name_str}{format}"
        else:
            file_path = f"{self.saving_dir}{self.job_id}_{name_str}{format}"
        
        with open(file_path, 'wb') as f:
            pickle.dump(samples, f)
            pickle.dump(misfit, f)

    def ukf(self, num_iterations=10, alpha=1e-3, beta=2, kappa=0, eps=0.001):
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
        self.sigma_d = 0.01
        self.process_noise_scale = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])

        x = self.q0.copy()
        L = self.n_params
        lam = alpha**2 * (L + kappa) - L
    
        # Initialize weights for mean and covariance
        Wm = np.full(2 * L + 1, 0.5 / (L + lam))
        Wm[0] = lam / (L + lam)
        Wc = Wm.copy()
        Wc[0] += 1 - alpha**2 + beta

        P = np.eye(L) * self.process_noise_scale
        y_obs = self.get_data_list_vector()
        R = np.eye(len(y_obs)) * self.sigma_d**2

        trajectory = [x.copy() * self.scaling]

        for i in range(num_iterations):
            print(f"UKF iteration {i+1} of {num_iterations}")
            
            # Generate Sigma Points
            sqrtP = cholesky((L + lam) * P)
            sigmas = np.vstack([x] + [x + sqrtP[j] for j in range(L)] + [x - sqrtP[j] for j in range(L)])
            
            # Predict observation for each sigma point
            Y = []
            for s in sigmas:
                mt0 = s.copy() * self.scaling
                mt0[3:] = mt0[3:] * 2
                y_s = self.compute_syn_shifted_vector(mt0)
                Y.append(y_s)
            Y = np.array(Y)
            y_mean = Wm @ Y
    
            # Compute Covariances
            P_yy = sum(Wc[j] * np.outer(Y[j] - y_mean, Y[j] - y_mean) for j in range(2 * L + 1)) + R
            P_xy = sum(Wc[j] * np.outer(sigmas[j] - x, Y[j] - y_mean) for j in range(2 * L + 1))
    
            # Kalman Gain
            K = P_xy @ np.linalg.inv(P_yy)
    
            # State and Covariance Update
            x_new = x + K @ (y_obs - y_mean)
            trajectory.append(x_new.copy() * self.scaling)
            
            if np.linalg.norm(x_new - x) < eps:
                print(f"✅ UKF converged at iteration {i+1} (Δx < {eps})")
                x = x_new
                break

            x = x_new
            P = P - K @ P_yy @ K.T
            P = 0.5 * (P + P.T)  # Ensure numerical symmetry and positivity
    
        else:
            print(f"⚠️ UKF did not converge within {num_iterations} iterations")
    
        return x * self.scaling, trajectory, P