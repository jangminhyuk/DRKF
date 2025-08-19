#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common_utils.py contains common utility functions used across the project.
"""

import numpy as np
import pickle
from scipy.linalg import solve_discrete_are

def save_data(path, data):
    """Save data to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def is_stabilizable(A, B, tol=1e-9):
    """Check if the pair (A, B) is stabilizable."""
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.hstack([eig * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_detectable(A, C, tol=1e-9):
    """Check if the pair (A, C) is detectable."""
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.vstack([eig * np.eye(n) - A, C])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_positive_definite(M, tol=1e-9):
    """Check if matrix M is positive definite."""
    if not np.allclose(M, M.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False

def enforce_positive_definiteness(Sigma, epsilon=1e-4):
    """Enforce positive definiteness of a matrix."""
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        Sigma += (epsilon - min_eig) * np.eye(Sigma.shape[0])
    return Sigma

def generate_desired_trajectory(T_total, Amp=5.0, slope=1.0, omega=0.5, dt=0.2):
    """Generate a sinusoidal desired trajectory."""
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    
    x_d = Amp * np.sin(omega * time)
    vx_d = Amp * omega * np.cos(omega * time)
    y_d = slope * time
    vy_d = slope * np.ones(time_steps)
    
    return np.vstack((x_d, vx_d, y_d, vy_d))

def compute_lqr_cost(result, Q_lqr, R_lqr, K_lqr, desired_traj):
    """Compute LQR cost for trajectory tracking."""
    x = result['state_traj']
    x_est = result['est_state_traj']
    T_steps = x.shape[0]
    cost = 0.0
    for t in range(T_steps):
        error = x[t] - desired_traj[:, t].reshape(-1, 1)
        u = -K_lqr @ (x_est[t] - desired_traj[:, t].reshape(-1, 1))
        cost += (error.T @ Q_lqr @ error)[0, 0] + (u.T @ R_lqr @ u)[0, 0]
    return cost