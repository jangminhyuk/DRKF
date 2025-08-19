#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_filter.py contains common functionality shared across all filter implementations.
"""

import numpy as np
import time

def generate_shared_noise_sequences(T, nx, ny, dist, noise_dist, 
                                  true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                                  x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                                  x0_scale=None, w_scale=None, v_scale=None, seed=42):
    """Generate shared noise sequences for consistent experiments across filters."""
    np.random.seed(seed)
    
    # Create temporary sampler for generating sequences
    A_temp, C_temp = np.eye(nx), np.eye(ny, nx)
    B_temp = np.eye(nx, 1)
    temp_params = np.zeros((nx, 1)), np.eye(nx)
    temp_params_y = np.zeros((ny, 1)), np.eye(ny)
    
    sampler = BaseFilter(T, dist, noise_dist, (A_temp, C_temp), B_temp,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        *temp_params, *temp_params, *temp_params_y,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale)
    
    # Generate sequences
    sequences = {}
    
    # Initial state
    sequences['x0'] = sampler.sample_initial_state()
    
    # Process and measurement noise sequences
    w_seq = np.zeros((T+1, nx, 1))
    v_seq = np.zeros((T+1, ny, 1))
    
    for t in range(T+1):
        w_seq[t] = sampler.sample_process_noise()
        v_seq[t] = sampler.sample_measurement_noise()
    
    sequences['w'] = w_seq
    sequences['v'] = v_seq
    
    return sequences

class BaseFilter:
    """Base class containing common distribution sampling and simulation logic."""
    
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None, shared_noise_sequences=None):
        
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.B = B
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        # True parameters
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_Sigma_v = true_Sigma_v
        
        # Nominal parameters
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_mu_v = nominal_mu_v
        self.nominal_Sigma_v = nominal_Sigma_v
        
        # Distribution bounds and scales
        if self.dist in ["uniform", "quadratic"]:
            self.x0_max, self.x0_min = x0_max, x0_min
            self.w_max, self.w_min = w_max, w_min
        if self.noise_dist in ["uniform", "quadratic"]:
            self.v_max, self.v_min = v_max, v_min
        if self.dist == "laplace":
            self.x0_scale, self.w_scale = x0_scale, w_scale
        if self.noise_dist == "laplace":
            self.v_scale = v_scale
            
        self.K_lqr = None
        self.shared_noise_sequences = shared_noise_sequences
        self._noise_index = 0
    
    def normal(self, mu, Sigma, N=1):
        return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T
    
    def uniform(self, a, b, N=1):
        n = a.shape[0]
        return a + (b - a) * np.random.rand(n, N)
    
    def quad_inverse(self, x, b, a):
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                beta = (a[j] + b[j]) / 2.0
                alpha = 12.0 / ((b[j] - a[j]) ** 3)
                tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
                if tmp >= 0:
                    x[i, j] = beta + tmp ** (1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp) ** (1.0/3.0)
        return x
    
    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(N, n).T
        return self.quad_inverse(x, max_val, min_val)
    
    def laplace(self, mu, scale, N=1):
        return np.random.laplace(mu[:, 0], scale, size=(N, mu.shape[0])).T
    
    def sample_initial_state(self):
        if self.shared_noise_sequences is not None:
            return self.shared_noise_sequences['x0']
        
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov)
        elif self.dist == "quadratic":
            return self.quadratic(self.x0_max, self.x0_min)
        elif self.dist == "laplace":
            return self.laplace(self.true_x0_mean, self.x0_scale)
        else:
            raise ValueError("Unsupported distribution for initial state.")
    
    def sample_process_noise(self):
        if self.shared_noise_sequences is not None:
            w = self.shared_noise_sequences['w'][self._noise_index]
            return w
        
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w)
        elif self.dist == "quadratic":
            return self.quadratic(self.w_max, self.w_min)
        elif self.dist == "laplace":
            return self.laplace(self.true_mu_w, self.w_scale)
        else:
            raise ValueError("Unsupported distribution for process noise.")
    
    def sample_measurement_noise(self):
        if self.shared_noise_sequences is not None:
            v = self.shared_noise_sequences['v'][self._noise_index]
            return v
        
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_Sigma_v)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min)
        elif self.noise_dist == "laplace":
            return self.laplace(self.true_mu_v, self.v_scale)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")
    
    def _run_simulation_loop(self, forward_method, desired_trajectory=None):
        """Common simulation loop for both forward and forward_track methods."""
        start_time = time.time()
        T, nx, ny, A, C, B = self.T, self.nx, self.ny, self.A, self.C, self.B
        
        # Allocate arrays
        x = np.zeros((T+1, nx, 1))
        y = np.zeros((T+1, ny, 1))
        x_est = np.zeros((T+1, nx, 1))
        mse = np.zeros(T+1)
        error = np.zeros((T+1, nx, 1)) if desired_trajectory is not None else None
        
        # Reset noise index for consistent sequences across experiments
        self._noise_index = 0
        
        # Initialization
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        
        # First measurement and update
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        x_est[0] = self._initial_update(x_est[0], y[0])
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # Main loop
        for t in range(T):
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            
            # Compute control
            if desired_trajectory is not None:
                desired = desired_trajectory[:, t].reshape(-1, 1)
                error[t] = x_est[t] - desired
                u = -self.K_lqr @ error[t]
            else:
                u = -self.K_lqr @ x_est[t]
            
            # State propagation
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Increment noise index after sampling process noise
            self._noise_index += 1
            
            # Measurement
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Filter update
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            x_est[t+1] = forward_method(x_pred, y[t+1], t+1)
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        result = {
            'comp_time': time.time() - start_time,
            'state_traj': x,
            'output_traj': y,
            'est_state_traj': x_est,
            'mse': mse
        }
        if error is not None:
            result['tracking_error'] = error
        return result
    
    def _initial_update(self, x_est_init, y0):
        """Override in subclasses for specific initial update logic."""
        raise NotImplementedError
    
    def forward(self):
        """Override in subclasses."""
        raise NotImplementedError
    
    def forward_track(self, desired_trajectory):
        """Override in subclasses."""
        raise NotImplementedError