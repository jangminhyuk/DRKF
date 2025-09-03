#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main5.py

This experiment runs a state estimation comparison without controller.
The system is:
    x_{t+1} = A x_t + w_t,    y_t = C x_t + v_t,
    
Multiple filters (KF, DRKF variants) are compared for state estimation accuracy
without any control input (B matrix and control are ignored).

Usage example:
    python main5.py --dist normal
    python main5.py --dist quadratic  
    python main5.py --dist laplace
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
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
from LQR_with_estimator.DRKF_neurips import DRKF_neurips
from common_utils import (save_data, is_stabilizable, is_detectable, is_positive_definite,
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
        x_true = A @ x_true + true_w  # No control input
        x_true_all[t+1] = x_true
    return x_true_all, y_all

# --- Modified Experiment Function ---
def run_experiment(exp_idx, dist, num_sim, seed_base, robust_val, T_total, filters_to_execute):
    np.random.seed(seed_base + exp_idx)
    
    # Set time horizon to T=50
    T = 50  # simulation horizon
    
    # Original system dynamics (commented out)
    # A = np.array([[1, dt, 0, 0],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, dt],
    #               [0, 0, 0, 1]])
    # B = np.zeros((4, 2))  # No control input
    # C = np.array([[1, 0, 0, 0],
    #               [0, 0, 1, 0]])
    
    # New random system with unstable A and detectable (A,C)
    nx = 4
    ny = 2
    
    # Generate random unstable A matrix
    np.random.seed(42)  # For reproducibility
    while True:
        A = np.random.randn(nx, nx) * 0.5
        eigenvals = np.linalg.eigvals(A)
        if np.max(np.real(eigenvals)) > 1.0:  # Check if unstable
            break
    
    # Generate random C matrix ensuring detectability
    C = np.random.randn(ny, nx) * 0.5
    
    # Check detectability condition
    while not is_detectable(A, C):
        C = np.random.randn(ny, nx) * 0.5
    
    B = np.zeros((nx, 2))  # No control input
    
    system_data = (A, C)
    
    nx = 4; nw = 4; nu = 2; ny = 2
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
    N_data = 10
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
    
    if not is_detectable(A, C):
        print("Warning: (A, C) is not detectable!")
        exit()
    if not is_positive_definite(nominal_Sigma_w):
        print("Warning: nominal_Sigma_w is not positive definite!")
        exit()
    if not is_positive_definite(nominal_Sigma_v):
        print("Warning: nominal_Sigma_v (noise covariance) is not positive definite!")
        exit()
    
    # --- Simulation Functions for Each Filter (No Control) ---
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
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
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
        res = estimator.forward()
        return res

    def run_simulation_drkf_neurips(sim_idx_local):
        estimator = DRKF_neurips(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
            theta=robust_val)
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
        res = estimator.forward()
        return res

    def run_simulation_mmse_baseline(sim_idx_local):
        # MMSE baseline using true distribution parameters (best possible estimator)
        estimator = KF(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            # Use TRUE parameters as nominal (perfect model knowledge)
            nominal_x0_mean=x0_mean, nominal_x0_cov=x0_cov,
            nominal_mu_w=mu_w, nominal_Sigma_w=Sigma_w,
            nominal_mu_v=mu_v, nominal_Sigma_v=Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
        estimator.K_lqr = np.zeros((nu, nx))  # Dummy LQR gain (no control)
        res = estimator.forward()
        return res

    
    # Filter execution mapping
    filter_functions = {
        'finite': run_simulation_finite,
        'inf': run_simulation_inf_kf,
        'drkf_inf': run_simulation_inf_drkf,
        'bcot': run_simulation_bcot,
        'drkf_finite': run_simulation_finite_drkf,
        'drkf_inf_cdc': run_simulation_inf_drkf_cdc,
        'drkf_finite_cdc': run_simulation_finite_drkf_cdc,
        'risk': run_simulation_risk,
        'drkf_neurips': run_simulation_drkf_neurips
    }
    
    # Execute only selected filters
    filter_results = {}
    for filter_name in filters_to_execute:
        if filter_name in filter_functions:
            results = [filter_functions[filter_name](i) for i in range(num_sim)]
            filter_results[filter_name] = {
                'results': results,
                'mse_mean': np.mean([np.mean(r['mse']) for r in results]),
                'rep_state': results[0]['state_traj']
            }
    
    # MMSE baseline (always executed)
    results_mmse_baseline = [run_simulation_mmse_baseline(i) for i in range(num_sim)]
    mse_mean_mmse_baseline = np.mean([np.mean(r['mse']) for r in results_mmse_baseline])
    rep_state_mmse_baseline = results_mmse_baseline[0]['state_traj']
    
    # Calculate regret (difference from MMSE baseline) for each executed filter
    overall_results = {
        'mmse_baseline': mse_mean_mmse_baseline,
        'mmse_baseline_state': rep_state_mmse_baseline,
    }
    
    for filter_name in filters_to_execute:
        if filter_name in filter_results:
            mse_mean = filter_results[filter_name]['mse_mean']
            rep_state = filter_results[filter_name]['rep_state']
            regret = mse_mean - mse_mean_mmse_baseline
            
            overall_results[filter_name] = mse_mean
            overall_results[f'{filter_name}_state'] = rep_state
            overall_results[f'{filter_name}_regret'] = regret
    # Return raw results for executed filters
    raw_results = {'mmse_baseline': results_mmse_baseline}
    for filter_name in filters_to_execute:
        if filter_name in filter_results:
            raw_results[filter_name] = filter_results[filter_name]['results']
    
    return overall_results, raw_results

# --- Main Routine ---
def main(dist, num_sim, num_exp, T_total):
    seed_base = 2024
    if dist=='normal':
        robust_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0]
    elif dist=='quadratic':
        robust_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0]
    elif dist=='laplace':
        robust_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Configurable filter execution list - modify this to enable/disable filters
    # Available filters: 'finite', 'inf', 'risk', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf'
    filters_to_execute = ['finite', 'inf', 'drkf_neurips', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    filters = filters_to_execute
    filter_labels = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'risk': "Risk-Sensitive Filter",
        'drkf_neurips': "DRKF (NeurIPS)",
        'bcot': "BCOT",
        'drkf_finite_cdc': "DRKF (ours, finite, CDC)",
        'drkf_inf_cdc': "DRKF (ours, inf, CDC)",
        'drkf_finite': "DRKF (ours, finite)",
        'drkf_inf': "DRKF (ours, inf)"
    }
    
    all_results = {}
    raw_experiments_data = {}   # Store raw experiments for each robust candidate.
    for robust_val in robust_vals:
        print(f"Running experiments for robust parameter = {robust_val}")
        experiments = Parallel(n_jobs=-1)(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, robust_val, T_total, filters_to_execute)
            for exp_idx in range(num_exp)
        )
        # Unpack overall results from the tuple returned by run_experiment.
        overall_experiments = [exp[0] for exp in experiments]
        all_mse = {key: [] for key in filters_to_execute}
        all_regret = {key: [] for key in filters_to_execute}
        mmse_baseline_values = []
        
        for exp in overall_experiments:
            # Collect MSE values for each filter
            for key in filters_to_execute:
                if key in exp:  # Only process if filter was executed
                    all_mse[key].append(np.mean(exp[key]))
                    all_regret[key].append(exp[f'{key}_regret'])
            # Collect MMSE baseline values
            mmse_baseline_values.append(exp['mmse_baseline'])
            
        final_mse = {key: np.mean(all_mse[key]) for key in filters_to_execute if all_mse[key]}
        final_mse_std = {key: np.std(all_mse[key]) for key in filters_to_execute if all_mse[key]}
        final_regret = {key: np.mean(all_regret[key]) for key in filters_to_execute if all_regret[key]}
        final_regret_std = {key: np.std(all_regret[key]) for key in filters_to_execute if all_regret[key]}
        mmse_baseline_mean = np.mean(mmse_baseline_values)
        mmse_baseline_std = np.std(mmse_baseline_values)
        
        # Store representative state trajectories from the first experiment run for this robust value
        rep_state = {filt: overall_experiments[0][f"{filt}_state"] for filt in filters_to_execute if f"{filt}_state" in overall_experiments[0]}
        all_results[robust_val] = {
            'mse': final_mse,
            'mse_std': final_mse_std,
            'regret': final_regret,
            'regret_std': final_regret_std,
            'mmse_baseline': mmse_baseline_mean,
            'mmse_baseline_std': mmse_baseline_std,
            'state': rep_state
        }
        # Save the raw experiments for this candidate robust parameter.
        raw_experiments_data[robust_val] = [exp[1] for exp in experiments]
        print(f"Candidate robust parameter {robust_val}: Average MSE = {final_mse}")
    
    optimal_results = {}
    optimal_regret_results = {}
    
    for f in filters_to_execute:
        if f in ['finite', 'inf']:
            candidate = list(all_results.values())[0]
            optimal_results[f] = {
                'robust_val': "N/A",
                'mse': candidate['mse'][f],
                'mse_std': candidate['mse_std'][f],
                'regret': candidate['regret'][f],
                'regret_std': candidate['regret_std'][f]
            }
            optimal_regret_results[f] = optimal_results[f].copy()
        else:
            # Find best MSE
            best_val_mse = None
            best_mse = np.inf
            for robust_val, res in all_results.items():
                current_mse = res['mse'][f]
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_val_mse = robust_val
            
            # Find best regret
            best_val_regret = None
            best_regret = np.inf
            for robust_val, res in all_results.items():
                current_regret = res['regret'][f]
                if current_regret < best_regret:
                    best_regret = current_regret
                    best_val_regret = robust_val
            
            optimal_results[f] = {
                'robust_val': best_val_mse,
                'mse': all_results[best_val_mse]['mse'][f],
                'mse_std': all_results[best_val_mse]['mse_std'][f],
                'regret': all_results[best_val_mse]['regret'][f],
                'regret_std': all_results[best_val_mse]['regret_std'][f]
            }
            
            optimal_regret_results[f] = {
                'robust_val': best_val_regret,
                'mse': all_results[best_val_regret]['mse'][f],
                'mse_std': all_results[best_val_regret]['mse_std'][f],
                'regret': all_results[best_val_regret]['regret'][f],
                'regret_std': all_results[best_val_regret]['regret_std'][f]
            }
            
        print(f"Optimal robust parameter for {f} (MSE): {optimal_results[f]['robust_val']}")
        print(f"Optimal robust parameter for {f} (Regret): {optimal_regret_results[f]['robust_val']}")
    
    sorted_optimal = sorted(optimal_results.items(), key=lambda item: item[1]['mse'])
    sorted_optimal_regret = sorted(optimal_regret_results.items(), key=lambda item: item[1]['regret'])
    
    print("\nSummary of Optimal Results (sorted by MSE):")
    for filt, info in sorted_optimal:
        print(f"{filt}: Optimal robust parameter = {info['robust_val']}, Average MSE = {info['mse']:.4f} ({info['mse_std']:.4f}), Regret = {info['regret']:.4f} ({info['regret_std']:.4f})")
    
    print("\nSummary of Optimal Results (sorted by Regret):")
    for filt, info in sorted_optimal_regret:
        print(f"{filt}: Optimal robust parameter = {info['robust_val']}, Average Regret = {info['regret']:.4f} ({info['regret_std']:.4f}), MSE = {info['mse']:.4f} ({info['mse_std']:.4f})")
    
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_data(os.path.join(results_path, f'overall_results_{dist}_estimation.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}_estimation.pkl'), optimal_results)
    save_data(os.path.join(results_path, f'optimal_regret_results_{dist}_estimation.pkl'), optimal_regret_results)
    # Save raw experiments data.
    save_data(os.path.join(results_path, f'raw_experiments_{dist}_estimation.pkl'), raw_experiments_data)
    print("State estimation experiments completed for all robust parameters.")
    
    # --- Print Readable Data for the User ---
    print("\nDetailed Experiment Results (MSE-optimized):")
    header = "{:<50} {:<35} {:<35} {:<15}".format("Method", "Average MSE", "Average Regret", "Best theta")
    print(header)
    print("-"*135)
    for filt in filters_to_execute:
        if filt in optimal_results:
            best_theta = optimal_results[filt]['robust_val']
        mse = optimal_results[filt]['mse']
        mse_std = optimal_results[filt]['mse_std']
        regret = optimal_results[filt]['regret']
        regret_std = optimal_results[filt]['regret_std']
        mse_str = f"{mse:.3f} ({mse_std:.3f})"
        regret_str = f"{regret:.3f} ({regret_std:.3f})"
        print("{:<50} {:<35} {:<35} {:<15}".format(filter_labels[filt], mse_str, regret_str, best_theta))
    
    print("\nDetailed Experiment Results (Regret-optimized):")
    header = "{:<50} {:<35} {:<35} {:<15}".format("Method", "Average MSE", "Average Regret", "Best theta")
    print(header)
    print("-"*135)
    for filt in filters_to_execute:
        if filt in optimal_regret_results:
            best_theta_regret = optimal_regret_results[filt]['robust_val']
        mse = optimal_regret_results[filt]['mse']
        mse_std = optimal_regret_results[filt]['mse_std']
        regret = optimal_regret_results[filt]['regret']
        regret_std = optimal_regret_results[filt]['regret_std']
        mse_str = f"{mse:.3f} ({mse_std:.3f})"
        regret_str = f"{regret:.3f} ({regret_std:.3f})"
        print("{:<50} {:<35} {:<35} {:<15}".format(filter_labels[filt], mse_str, regret_str, best_theta_regret))
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal, quadratic, or laplace)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=20, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--time', default=5, type=int,
                        help="Total simulation time")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.time)