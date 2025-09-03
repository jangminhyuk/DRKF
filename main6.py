#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main6.py

This experiment generates 3D data for average regret MSE with respect to theta_w and theta_v.
The system is:
    x_{t+1} = A x_t + w_t,    y_t = C x_t + v_t,
    
Generates regret MSE data for different theta_w and theta_v combinations
for future 3D plotting (theta_w vs theta_v vs average regret MSE).

Usage example:
    python main6.py --dist normal
    python main6.py --dist quadratic  
    python main6.py --dist laplace
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
import pandas as pd

# Import filter implementations.
from LQR_with_estimator.KF import KF
from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
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
def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, Sigma_v,
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
            true_v = normal(mu_v, Sigma_v)
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
def run_experiment(exp_idx, dist, num_sim, seed_base, theta_w, theta_v, T_total):
    np.random.seed(seed_base + exp_idx)
    
    # Set time horizon to T=50
    T = 50  # simulation horizon
    
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
    
    # --- Simulation Functions for Standard KF and DRKF ---
    def run_simulation_standard_kf(sim_idx_local):
        estimator = KF(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=x0_mean, nominal_x0_cov=x0_cov,
            nominal_mu_w=mu_w, nominal_Sigma_w=Sigma_w,
            nominal_mu_v=mu_v, nominal_Sigma_v=Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale)
        estimator.K_lqr = np.zeros((nu, nx))
        res = estimator.forward()
        return res

    def run_simulation_drkf_ours_inf(sim_idx_local):
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
            theta_w=theta_w, theta_v=theta_v)
        estimator.K_lqr = np.zeros((nu, nx))
        res = estimator.forward()
        return res
    
    # Run standard KF baseline (using true parameters)
    results_standard_kf = [run_simulation_standard_kf(i) for i in range(num_sim)]
    mse_mean_standard_kf = np.mean([np.mean(r['mse']) for r in results_standard_kf])
    
    # Run DRKF_ours_inf
    results_drkf = [run_simulation_drkf_ours_inf(i) for i in range(num_sim)]
    mse_mean_drkf = np.mean([np.mean(r['mse']) for r in results_drkf])
    
    # Calculate regret (difference from standard KF baseline)
    regret_drkf = mse_mean_drkf - mse_mean_standard_kf
    
    return {
        'theta_w': theta_w,
        'theta_v': theta_v,
        'standard_kf_mse': mse_mean_standard_kf,
        'drkf_mse': mse_mean_drkf,
        'regret': regret_drkf
    }

# --- Main Routine ---
def main(dist, num_sim, num_exp, T_total):
    seed_base = 2024
    
    # Define theta_w and theta_v ranges for 3D plotting - denser grid for smoother plots
    theta_w_vals = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    theta_v_vals = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    
    # Storage for 3D plotting data
    regret_data = []
    
    total_combinations = len(theta_w_vals) * len(theta_v_vals)
    combination_count = 0
    
    for theta_w in theta_w_vals:
        for theta_v in theta_v_vals:
            combination_count += 1
            print(f"Running experiments for theta_w={theta_w}, theta_v={theta_v} ({combination_count}/{total_combinations})")
            
            experiments = Parallel(n_jobs=-1)(
                delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, theta_w, theta_v, T_total)
                for exp_idx in range(num_exp)
            )
            
            # Calculate average regret across all experiments
            regrets = [exp['regret'] for exp in experiments]
            avg_regret = np.mean(regrets)
            std_regret = np.std(regrets)
            
            regret_data.append({
                'theta_w': theta_w,
                'theta_v': theta_v,
                'avg_regret': avg_regret,
                'std_regret': std_regret
            })
            
            print(f"  Average regret: {avg_regret:.6f} ± {std_regret:.6f}")
    
    # Save data for 3D plotting
    results_path = "./results/3d_data/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Save as pickle for easy loading
    save_data(os.path.join(results_path, f'regret_3d_data_{dist}.pkl'), regret_data)
    
    # Also save as CSV for other tools
    df = pd.DataFrame(regret_data)
    df.to_csv(os.path.join(results_path, f'regret_3d_data_{dist}.csv'), index=False)
    
    print(f"\n3D plotting data saved to {results_path}")
    print(f"Generated {len(regret_data)} data points for theta_w x theta_v combinations")
    
    return regret_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal, quadratic, or laplace)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=50, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--time', default=5, type=int,
                        help="Total simulation time")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.time)