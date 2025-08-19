#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main3.py

This experiment runs an open–loop simulation to test estimator performance WITHOUT control:
    xₜ₊₁ = A xₜ + wₜ,    yₜ = C xₜ + vₜ,
    
The nominal parameters are obtained via EM (covariances only) while the known mean
vectors (x0_mean, mu_w, mu_v) are used for the means.
Then, seven filters (KF, KF_inf, DRKF, DRKF_CDC, DRKF_finite_CDC, BCOT, and risk–sensitive filter)
(from the folder LQR_with_estimator) are run in open–loop WITHOUT any controller
for each candidate robust parameter value.
For each simulation run, only the mean squared error (MSE) trajectory is computed
and returned. For each filter the candidate robust parameter with the lowest average
MSE is chosen as "optimal." Finally, the MSE performance of each filter using its
optimal robust parameter is saved for comparison.
    
Usage example:
    python main3.py --dist normal
    python main3.py --dist quadratic
    python main3.py --dist laplace
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Import filter implementations.
from LQR_with_estimator.KF import KF
from LQR_with_estimator.KF_inf import KF_inf
from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
from LQR_with_estimator.DRKF_ours_finite import DRKF_ours_finite
from LQR_with_estimator.DRKF_ours_inf_CDC import DRKF_ours_inf_CDC
from LQR_with_estimator.DRKF_ours_finite_CDC import DRKF_ours_finite_CDC
from LQR_with_estimator.BCOT import BCOT 
from LQR_with_estimator.risk_sensitive import RiskSensitive
from common_utils import (save_data, is_detectable, is_positive_definite,
                         enforce_positive_definiteness)

# Import distribution functions from base filter
from LQR_with_estimator.base_filter import BaseFilter

# Create a helper instance for distribution sampling - temporary instance just for sampling functions
_temp_A, _temp_C = np.eye(2), np.eye(2)
_temp_params = np.zeros((2, 1)), np.eye(2)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(2, 1),
                     *_temp_params, *_temp_params, *_temp_params,
                     *_temp_params, *_temp_params, *_temp_params)

def normal(mu, Sigma, N=1):
    return _sampler.normal(mu, Sigma, N)

def uniform(a, b, N=1):
    return _sampler.uniform(a, b, N).T if N > 1 else _sampler.uniform(a, b, N)

def quadratic(w_max, w_min, N=1):
    result = _sampler.quadratic(w_max, w_min, N)
    return result.T if N > 1 else result

def laplace(mu, scale, N=1):
    return _sampler.laplace(mu, scale, N)

# --- True Data Generation ---
def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist,
                  x0_scale=None, w_scale=None, v_scale=None):
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    
    if dist == "normal":
        x_true = normal(x0_mean, x0_cov)
    elif dist == "quadratic":
        x_true = quadratic(x0_max, x0_min)
    elif dist == "laplace":
        x_true = laplace(x0_mean, x0_scale)
    x_true_all[0] = x_true
    
    for t in range(T):
        if dist == "normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist == "quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)
        elif dist == "laplace":
            true_w = laplace(mu_w, w_scale)
            true_v = laplace(mu_v, v_scale)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    return x_true_all, y_all


# --- Open-Loop System (No Controller Functions Needed) ---

# --- Open-Loop Experiment Function (No Controller) ---
def run_experiment(exp_idx, dist, num_sim, seed_base, robust_val, T_total):
    np.random.seed(seed_base + exp_idx)
    
    dt = 0.2
    time_steps = int(T_total / dt) + 1
    T = time_steps - 1  # simulation horizon
    
    # Discrete-time system dynamics (4D double integrator) - NO CONTROL
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    C = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    
    system_data = (A, C)
    
    nx = 4; nw = 4; ny = 2
    B = np.zeros((nx, 2))  # Dummy control matrix (not used in open-loop)
    if dist == "normal":
        mu_w = 0.0 * np.ones((nw, 1))
        Sigma_w = 0.01 * np.eye(nw)
        x0_mean = 0.0 * np.ones((nx, 1))
        x0_cov = 0.01 * np.eye(nx)
        mu_v = 0.0 * np.ones((ny, 1))
        Sigma_v = 0.01 * np.eye(ny)
        v_max = v_min = None
        w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    elif dist == "quadratic":
        w_max = 0.1 * np.ones(nx)
        w_min = -0.1 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[:, None]
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)
        x0_max = 0.1 * np.ones(nx)
        x0_min = -0.1 * np.ones(nx)
        x0_mean = (0.5 * (x0_max + x0_min))[:, None]
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)
        v_min = -0.1 * np.ones(ny)
        v_max = 0.1 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)
        x0_scale = w_scale = v_scale = None
    elif dist == "laplace":
        mu_w = 0.0 * np.ones((nw, 1))
        w_scale = 0.1 * np.ones(nw)
        Sigma_w = 2.0 * (w_scale**2) * np.eye(nw)  # Laplace variance = 2*scale^2
        x0_mean = 0.0 * np.ones((nx, 1))
        x0_scale = 0.1 * np.ones(nx)
        x0_cov = 2.0 * (x0_scale**2) * np.eye(nx)
        mu_v = 0.0 * np.ones((ny, 1))
        v_scale = 0.1 * np.ones(ny)
        Sigma_v = 2.0 * (v_scale**2) * np.eye(ny)
        v_max = v_min = None
        w_max = w_min = x0_max = x0_min = None
    else:
        raise ValueError("Unsupported noise distribution.")
    
    # --- Generate Data for EM ---
    N_data = 5
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, Sigma_v,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist,
                                x0_scale, w_scale, v_scale)
    y_all_em = y_all_em.squeeze()
    
    # --- EM Estimation of Nominal Covariances ---
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    kf_em = KalmanFilter(transition_matrices=A,
                           observation_matrices=C,
                           transition_covariance=Sigma_w_hat,
                           observation_covariance=Sigma_v_hat,
                           transition_offsets=mu_w_hat.squeeze(),
                           observation_offsets=mu_v_hat.squeeze(),
                           initial_state_mean=mu_x0_hat.squeeze(),
                           initial_state_covariance=Sigma_x0_hat,
                           em_vars=['transition_covariance', 'observation_covariance',
                                    'transition_offsets', 'observation_offsets'])
    max_iter = 100
    eps_log = 1e-4
    loglikelihoods = np.zeros(max_iter)
    for i in range(max_iter):
        kf_em = kf_em.em(X=y_all_em, n_iter=1)
        loglikelihoods[i] = kf_em.loglikelihood(y_all_em)
        Sigma_w_hat = kf_em.transition_covariance
        Sigma_v_hat = kf_em.observation_covariance
        mu_x0_hat = kf_em.initial_state_mean
        Sigma_x0_hat = kf_em.initial_state_covariance
        if i > 0 and (loglikelihoods[i] - loglikelihoods[i-1] <= eps_log):
            break
    
    Sigma_w_hat = enforce_positive_definiteness(Sigma_w_hat)
    Sigma_v_hat = enforce_positive_definiteness(Sigma_v_hat)
    Sigma_x0_hat = enforce_positive_definiteness(Sigma_x0_hat)
    
    nominal_mu_w    = mu_w
    nominal_Sigma_w = Sigma_w_hat.copy()
    nominal_mu_v    = mu_v
    nominal_Sigma_v = Sigma_v_hat.copy()
    nominal_x0_mean = x0_mean
    nominal_x0_cov  = Sigma_x0_hat.copy()
    
    # --- System Checks (No LQR in Open-Loop) ---
    if not is_detectable(A, C):
        print("Warning: (A, C) is not detectable!")
        exit()
    if not is_positive_definite(nominal_Sigma_w):
        print("Warning: nominal_Sigma_w is not positive definite!")
        exit()
    if not is_positive_definite(nominal_Sigma_v):
        print("Warning: nominal_Sigma_v (noise covariance) is not positive definite!")
        exit()
    
    # --- Dummy LQR Gain for Open-Loop (Zero Control) ---
    K_lqr = np.zeros((2, nx))  # Zero control input for open-loop
    
    # --- Simulation Functions for Each Filter (Open-Loop) ---
    def run_simulation_finite(sim_idx_local):
        estimator = KF(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res

    def run_simulation_inf_kf(sim_idx_local):
        estimator = KF_inf(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res

    def run_simulation_inf_drkf(sim_idx_local):
        estimator = DRKF_ours_inf(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_w=robust_val, theta_v=robust_val)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res
    
    def run_simulation_finite_drkf(sim_idx_local):
        estimator = DRKF_ours_finite(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_x=robust_val, theta_v=robust_val, theta_w=robust_val)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res

    def run_simulation_inf_drkf_cdc(sim_idx_local):
        estimator = DRKF_ours_inf_CDC(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_x=robust_val, theta_v=robust_val)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res
    
    def run_simulation_finite_drkf_cdc(sim_idx_local):
        estimator = DRKF_ours_finite_CDC(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta_x=robust_val, theta_v=robust_val)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res

    def run_simulation_bcot(sim_idx_local):
        estimator = BCOT(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            radius=robust_val, maxit=20,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res

    def run_simulation_risk(sim_idx_local):
        estimator = RiskSensitive(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=x0_mean,  # known initial state mean
            nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=mu_w,        # known process noise mean
            nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=mu_v,        # known measurement noise mean
            nominal_Sigma_v=nominal_Sigma_v,
            theta_rs=robust_val,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
        estimator.K_lqr = K_lqr  # Assign dummy zero controller
        res = estimator.forward()
        return res

    results_finite = [run_simulation_finite(i) for i in range(num_sim)]
    mse_mean_finite = np.mean([np.mean(r['mse']) for r in results_finite])
    rep_state_finite = results_finite[0]['state_traj']

    results_inf = [run_simulation_inf_kf(i) for i in range(num_sim)]
    mse_mean_inf = np.mean([np.mean(r['mse']) for r in results_inf])
    rep_state_inf = results_inf[0]['state_traj']

    results_drkf = [run_simulation_inf_drkf(i) for i in range(num_sim)]
    mse_mean_drkf = np.mean([np.mean(r['mse']) for r in results_drkf])
    rep_state_drkf = results_drkf[0]['state_traj']

    results_bcot = [run_simulation_bcot(i) for i in range(num_sim)]
    mse_mean_bcot = np.mean([np.mean(r['mse']) for r in results_bcot])
    rep_state_bcot = results_bcot[0]['state_traj']

    results_risk = [run_simulation_risk(i) for i in range(num_sim)]
    mse_mean_risk = np.mean([np.mean(r['mse']) for r in results_risk])
    rep_state_risk = results_risk[0]['state_traj']

    results_drkf_finite = [run_simulation_finite_drkf(i) for i in range(num_sim)]
    mse_mean_drkf_finite = np.mean([np.mean(r['mse']) for r in results_drkf_finite])
    rep_state_drkf_finite = results_drkf_finite[0]['state_traj']

    results_drkf_inf_cdc = [run_simulation_inf_drkf_cdc(i) for i in range(num_sim)]
    mse_mean_drkf_inf_cdc = np.mean([np.mean(r['mse']) for r in results_drkf_inf_cdc])
    rep_state_drkf_inf_cdc = results_drkf_inf_cdc[0]['state_traj']

    results_drkf_finite_cdc = [run_simulation_finite_drkf_cdc(i) for i in range(num_sim)]
    mse_mean_drkf_finite_cdc = np.mean([np.mean(r['mse']) for r in results_drkf_finite_cdc])
    rep_state_drkf_finite_cdc = results_drkf_finite_cdc[0]['state_traj']
    
    overall_results = {
        'finite': mse_mean_finite,
        'finite_state': rep_state_finite,
        'inf': mse_mean_inf,
        'inf_state': rep_state_inf,
        'drkf_inf': mse_mean_drkf,
        'drkf_inf_state': rep_state_drkf,
        'drkf_finite': mse_mean_drkf_finite,
        'drkf_finite_state': rep_state_drkf_finite,
        'drkf_inf_cdc': mse_mean_drkf_inf_cdc,
        'drkf_inf_cdc_state': rep_state_drkf_inf_cdc,
        'drkf_finite_cdc': mse_mean_drkf_finite_cdc,
        'drkf_finite_cdc_state': rep_state_drkf_finite_cdc,
        'bcot': mse_mean_bcot,
        'bcot_state': rep_state_bcot,
        'risk': mse_mean_risk,
        'risk_state': rep_state_risk
    }
    return overall_results, {
        'finite': results_finite,
        'inf': results_inf,
        'drkf_inf': results_drkf,
        'drkf_finite': results_drkf_finite,
        'drkf_inf_cdc': results_drkf_inf_cdc,
        'drkf_finite_cdc': results_drkf_finite_cdc,
        'bcot': results_bcot,
        'risk': results_risk
    }

# --- Main Routine ---
def main(dist, num_sim, num_exp, T_total):
    seed_base = 2024
    if dist=='normal':
        robust_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]
    elif dist=='quadratic':
        robust_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]
    elif dist=='laplace':
        robust_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]
    # Define filters (the same keys are used for MSE and state trajectories)
    filters = ['finite', 'inf', 'drkf_inf', 'drkf_finite', 'drkf_inf_cdc', 'drkf_finite_cdc', 'bcot', 'risk']
    filter_labels = {
        'finite': "Time-varying Kalman filter",
        'inf': "Time-invariant Kalman filter",
        'bcot': "BCOT filter",
        'risk': "Risk-Sensitive filter",
        'drkf_inf': "DR Kalman filter (ours, inf)",
        'drkf_finite': "DR Kalman filter (ours, finite)",
        'drkf_inf_cdc': "DR Kalman filter (ours, inf, CDC version)",
        'drkf_finite_cdc': "DR Kalman filter (ours, finite, CDC version)"
    }
    
    all_results = {}
    raw_experiments_data = {}   # Store raw experiments for each robust candidate.
    for robust_val in robust_vals:
        print(f"Running experiments for robust parameter = {robust_val}")
        experiments = Parallel(n_jobs=-1)(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, robust_val, T_total)
            for exp_idx in range(num_exp)
        )
        # Unpack overall results from the tuple returned by run_experiment.
        overall_experiments = [exp[0] for exp in experiments]
        mse_keys = filters
        all_mse = {key: [] for key in mse_keys}
        for exp in overall_experiments:
            for key in mse_keys:
                all_mse[key].append(np.mean(exp[key]))
        final_mse = {key: np.mean(all_mse[key]) for key in mse_keys}
        final_mse_std = {key: np.std(all_mse[key]) for key in mse_keys}
        # Store representative state trajectories from the first experiment run for this robust value
        rep_state = {filt: overall_experiments[0][f"{filt}_state"] for filt in filters}
        all_results[robust_val] = {
            'mse': final_mse,
            'mse_std': final_mse_std,
            'state': rep_state
        }
        # Save the raw experiments for this candidate robust parameter.
        raw_experiments_data[robust_val] = [exp[1] for exp in experiments]
        print(f"Candidate robust parameter {robust_val}: Average MSE = {final_mse}")
    
    optimal_results = {}
    for f in filters:
        if f in ['finite', 'inf']:
            candidate = list(all_results.values())[0]
            optimal_results[f] = {
                'robust_val': "N/A",
                'mse': candidate['mse'][f],
                'mse_std': candidate['mse_std'][f]
            }
        else:
            best_val = None
            best_mse = np.inf
            for robust_val, res in all_results.items():
                current_mse = res['mse'][f]
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_val = robust_val
            optimal_results[f] = {
                'robust_val': best_val,
                'mse': all_results[best_val]['mse'][f],
                'mse_std': all_results[best_val]['mse_std'][f]
            }
        print(f"Optimal robust parameter for {f}: {optimal_results[f]['robust_val']}")
    
    sorted_optimal = sorted(optimal_results.items(), key=lambda item: item[1]['mse'])
    print("\nSummary of Optimal Results (sorted by MSE):")
    for filt, info in sorted_optimal:
        print(f"{filt}: Optimal robust parameter = {info['robust_val']}, Average MSE = {info['mse']:.4f} ({info['mse_std']:.4f})")
    
    results_path = "./results/estimator/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_data(os.path.join(results_path, f'overall_results_{dist}_{dist}_openloop.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}_{dist}_openloop.pkl'), optimal_results)
    # Save raw experiments data.
    save_data(os.path.join(results_path, f'raw_experiments_{dist}_{dist}_openloop.pkl'), raw_experiments_data)
    print("Open-loop state estimation experiments completed for all robust parameters.")
    
    # --- Print MSE Results Only ---
    print("\nFinal MSE Results (Open-Loop Estimation):")
    header = "{:<50} {:<35} {:<15}".format("Method", "Average MSE", "Best theta")
    print(header)
    print("-"*100)
    for filt in filters:
        best_theta = optimal_results[filt]['robust_val']
        mse = optimal_results[filt]['mse']
        mse_std = optimal_results[filt]['mse_std']
        mse_str = f"{mse:.4f} ({mse_std:.4f})"
        print("{:<50} {:<35} {:<15}".format(filter_labels[filt], mse_str, best_theta))
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal, quadratic, or laplace)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=20, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--time', default=10, type=int,
                        help="Total simulation time")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.time)
