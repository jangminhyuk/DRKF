#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Ellipse Visualization of KF Sandwich Property

This script visualizes the KF sandwich property using 95% confidence ellipses
for LOW-KF, DRKF, and HIGH-KF posterior covariances at selected time steps.

The ellipses should show nested inclusion: LOW ⊆ DRKF ⊆ HIGH (visually).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
from drkf_tube_check import (
    drkf_spectral_verification, generate_random_pd_matrix,
    check_loewner_order
)

# Set professional matplotlib parameters with larger fonts
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2
})

def compute_ellipse_params(Sigma, confidence_level=0.95):
    """
    Compute ellipse parameters for plotting 95% confidence ellipse.
    
    Parameters:
    -----------
    Sigma : ndarray (2x2)
        Covariance matrix
    confidence_level : float
        Confidence level (default 0.95 for 95%)
        
    Returns:
    --------
    width : float
        Ellipse width (2 * sqrt(lambda1 * chi2))
    height : float
        Ellipse height (2 * sqrt(lambda2 * chi2))
    angle : float
        Ellipse rotation angle in degrees
    """
    # Chi-squared value for 95% confidence in 2D
    if confidence_level == 0.95:
        chi2_val = 5.991  # chi^2(2, 0.95)
    else:
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence_level, df=2)
    
    # Eigendecomposition
    eigenvals, eigenvecs = eigh(Sigma)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Compute ellipse dimensions
    width = 2 * np.sqrt(eigenvals[0] * chi2_val)
    height = 2 * np.sqrt(eigenvals[1] * chi2_val)
    
    # Compute rotation angle (in degrees)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    return width, height, angle

def plot_ellipse(ax, Sigma, center=(0, 0), style_dict=None, label=None):
    """
    Plot confidence ellipse on given axes.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    Sigma : ndarray (2x2)
        Covariance matrix
    center : tuple
        Ellipse center (x, y)
    style_dict : dict
        Style parameters (color, linestyle, linewidth, etc.)
    label : str
        Legend label
    """
    if style_dict is None:
        style_dict = {}
    
    width, height, angle = compute_ellipse_params(Sigma)
    
    ellipse = Ellipse(
        center, width, height, angle=angle,
        fill=False, 
        **style_dict
    )
    
    ax.add_patch(ellipse)
    
    # Add invisible line for legend
    if label:
        ax.plot([], [], label=label, **{k: v for k, v in style_dict.items() 
                                       if k in ['color', 'linestyle', 'linewidth']})
    
    return ellipse

def plot_3d_tube(low_posterior, drkf_posterior, high_posterior, T):
    """
    Plot 3D tube visualization showing posterior covariance ellipses evolution over time.
    
    This creates "tubes" by plotting 95% confidence ellipses of posterior covariances
    at different time steps and connecting them to form surfaces.
    
    - X-axis: Time steps (t)
    - Y-axis: x₁ (first state variable) 
    - Z-axis: x₂ (second state variable)
    
    Each ellipse cross-section represents the 95% confidence region of the
    posterior state estimate at that time step.
    
    Parameters:
    -----------
    low_posterior : list
        LOW-KF posterior covariances Σ_LOW(t)
    drkf_posterior : list
        DRKF posterior covariances Σ_DRKF(t)
    high_posterior : list
        HIGH-KF posterior covariances Σ_HIGH(t)
    T : int
        Time horizon
    """
    # === ADJUSTABLE PARAMETERS ===
    # Legend position - adjust these values to move the legend
    legend_x = 0.75  # Horizontal position (0=left, 1=right)
    legend_y = 0.85  # Vertical position (0=bottom, 1=top)
    legend_spacing = 0.8  # Line spacing in legend (smaller = tighter)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time steps to sample for tube construction (starting from t=1)
    time_steps = np.arange(1, T + 1, 2)  # Sample every 2 time steps starting from t=1
    
    # Parameters for ellipse discretization
    n_points = 20  # Number of points for ellipse circumference
    theta_circle = np.linspace(0, 2*np.pi, n_points)
    
    # Function to get ellipse points in 3D
    def get_ellipse_3d(Sigma, t_val, confidence_level=0.95):
        # Chi-squared value for 95% confidence in 2D
        chi2_val = 5.991
        
        # Eigendecomposition
        eigenvals, eigenvecs = eigh(Sigma)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Semi-axes lengths
        a = np.sqrt(eigenvals[0] * chi2_val)
        b = np.sqrt(eigenvals[1] * chi2_val)
        
        # Parametric ellipse in standard position
        x_std = a * np.cos(theta_circle)
        y_std = b * np.sin(theta_circle)
        
        # Rotate according to eigenvectors
        ellipse_points = np.array([x_std, y_std])
        rotated_points = eigenvecs @ ellipse_points
        
        # Add time dimension
        x_3d = rotated_points[0, :]
        y_3d = rotated_points[1, :]
        t_3d = np.full_like(x_3d, t_val)
        
        return t_3d, x_3d, y_3d
    
    # Generate tube surfaces
    all_low_t, all_low_x, all_low_y = [], [], []
    all_drkf_t, all_drkf_x, all_drkf_y = [], [], []
    all_high_t, all_high_x, all_high_y = [], [], []
    
    for t in time_steps:
        # Get ellipse points for each filter
        t_low, x_low, y_low = get_ellipse_3d(low_posterior[t], t)
        t_drkf, x_drkf, y_drkf = get_ellipse_3d(drkf_posterior[t], t)
        t_high, x_high, y_high = get_ellipse_3d(high_posterior[t], t)
        
        all_low_t.append(t_low)
        all_low_x.append(x_low)
        all_low_y.append(y_low)
        
        all_drkf_t.append(t_drkf)
        all_drkf_x.append(x_drkf)
        all_drkf_y.append(y_drkf)
        
        all_high_t.append(t_high)
        all_high_x.append(x_high)
        all_high_y.append(y_high)
    
    # Convert to arrays for surface plotting
    low_t = np.array(all_low_t)
    low_x = np.array(all_low_x)
    low_y = np.array(all_low_y)
    
    drkf_t = np.array(all_drkf_t)
    drkf_x = np.array(all_drkf_x)
    drkf_y = np.array(all_drkf_y)
    
    high_t = np.array(all_high_t)
    high_x = np.array(all_high_x)
    high_y = np.array(all_high_y)
    
    # Plot tube surfaces with better visibility for nesting
    # HIGH-KF (outermost) - Red with low alpha
    ax.plot_surface(high_t, high_x, high_y, alpha=0.2, color='red', label='HIGH-KF')
    
    # DRKF (middle) - Green with medium alpha  
    ax.plot_surface(drkf_t, drkf_x, drkf_y, alpha=0.7, color='green', label='DRKF')
    
    # LOW-KF (innermost) - Blue with higher alpha
    ax.plot_surface(low_t, low_x, low_y, alpha=0.4, color='blue', label='LOW-KF')
    
    # Add center lines for better visualization
    center_t = [t for t in time_steps]
    center_x = [0 for _ in time_steps]
    center_y = [0 for _ in time_steps]
    
    ax.plot(center_t, center_x, center_y, linewidth=2)
    
    # Set labels and title with more spacing from axes
    ax.set_xlabel('$t$', fontsize=20, labelpad=15)
    ax.set_ylabel('$x_1$', fontsize=20, labelpad=15)
    ax.set_zlabel('$x_2$', fontsize=20, labelpad=15)
    # ax.set_title('3D Tube Visualization: Posterior Covariance Evolution\n' + 
    #              r'95% Confidence Ellipses over Time, $\theta_w=0.05$, $\theta_v=0.05$', fontsize=18)
    
    # Create custom legend with colored patches (no line styles for 3D surfaces)
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black', label='LOW-KF'),
                      Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', label='DRKF'),
                      Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='HIGH-KF')]
    
    # Position legend using adjustable parameters
    legend = ax.legend(handles=legend_elements, 
                      bbox_to_anchor=(legend_x, legend_y), 
                      loc='upper left',
                      fontsize=18,
                      frameon=True,
                      fancybox=True,
                      shadow=False,
                      framealpha=0.9,
                      edgecolor='black',
                      handlelength=1.0,  # Size of legend patches (standard size)
                      handletextpad=0.5,  # Padding between patch and text
                      columnspacing=0.5,  # Space between columns
                      borderpad=0.8)      # Padding inside legend box (larger = bigger box)
    
    # Set frame linewidth separately
    legend.get_frame().set_linewidth(0.5)
    
    # Adjust alignment and spacing in legend
    legend.set_title(None)
    for text in legend.get_texts():
        text.set_verticalalignment('center')
    
    # Note: Rectangle patches should now display properly with standard dimensions
    
    # Make legend more compact vertically
    legend._legend_box.sep = legend_spacing
    
    # Set viewing angle for better visualization
    ax.view_init(elev=15, azim=-50)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure with extra padding to prevent cropping
    plt.tight_layout()
    plt.savefig('tube_3D.pdf', bbox_inches='tight', pad_inches=0.5)
    print(f"3D tube figure saved as 'tube_3D.pdf'")
    
    return fig

def main():
    print("2D Ellipse Visualization of KF Sandwich Property")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define 2D system
    nx, ny = 2, 2
    T = 40
    
    # Create more stable 2D system (eigenvalues closer to 1 for slower evolution)
    A = np.array([[0.98, 0.05], 
                  [0.02, 0.97]])
    C = np.array([[1.0, 0.1], 
                  [0.05, 1.0]])
    
    print(f"System dimensions: nx={nx}, ny={ny}")
    print(f"Time horizon: T={T}")
    print(f"A matrix eigenvalues: {np.linalg.eigvals(A)}")
    
    # Generate smaller nominal covariances for slower evolution
    nominal_Sigma_w = np.array([[0.05, 0.01], 
                                [0.01, 0.03]])
    nominal_Sigma_v = np.array([[0.04, 0.005], 
                                [0.005, 0.02]])
    nominal_x0_posterior = np.array([[0.008, 0.0015], 
                                     [0.0015, 0.005]])
    
    # Ensure positive definiteness
    from common_utils import is_positive_definite, enforce_positive_definiteness
    nominal_Sigma_w = enforce_positive_definiteness(nominal_Sigma_w)
    nominal_Sigma_v = enforce_positive_definiteness(nominal_Sigma_v)
    nominal_x0_posterior = enforce_positive_definiteness(nominal_x0_posterior)
    
    print(f"Process noise eigenvalues: {np.linalg.eigvals(nominal_Sigma_w)}")
    print(f"Measurement noise eigenvalues: {np.linalg.eigvals(nominal_Sigma_v)}")
    print(f"Initial posterior eigenvalues: {np.linalg.eigvals(nominal_x0_posterior)}")
    
    # Verify matrices are positive definite
    print(f"Process noise PD: {is_positive_definite(nominal_Sigma_w)}")
    print(f"Measurement noise PD: {is_positive_definite(nominal_Sigma_v)}")
    print(f"Initial posterior PD: {is_positive_definite(nominal_x0_posterior)}")
    
    # Wasserstein radii
    theta_w = 0.05
    theta_v = 0.05
    
    print(f"Wasserstein radii: θ_w={theta_w}, θ_v={theta_v}")
    print()
    
    # Create time-varying trajectories (constant in this example)
    nominal_Sigma_w_traj = [nominal_Sigma_w] * (T + 1)
    nominal_Sigma_v_traj = [nominal_Sigma_v] * (T + 1)
    
    # Run verification
    print("Running DRKF spectral verification...")
    results = drkf_spectral_verification(
        A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj,
        theta_w, theta_v, nominal_x0_posterior, T,
        verbose=False
    )
    
    # Extract covariance trajectories
    low_posterior = results['low_kf']['Sigma_posterior']
    high_posterior = results['high_kf']['Sigma_posterior']
    drkf_posterior = results['drkf']['Sigma_x_posterior']
    
    print("Verification completed successfully!")
    print()
    
    # Time points to visualize
    Tview = [1, 5, 20, 40]
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Style definitions
    low_style = {'color': 'blue', 'linestyle': '--', 'linewidth': 2}
    drkf_style = {'color': 'green', 'linestyle': '-', 'linewidth': 3}
    high_style = {'color': 'red', 'linestyle': ':', 'linewidth': 2}
    
    # Calculate global axis limits for consistent scaling across all plots
    global_limit = 0
    for t in Tview:
        Sigma_high = high_posterior[t]
        max_width, max_height, _ = compute_ellipse_params(Sigma_high)
        limit = max(max_width, max_height) * 0.6
        global_limit = max(global_limit, limit)
    
    for i, t in enumerate(Tview):
        ax = axes[i]
        
        # Get covariances at time t
        Sigma_low = low_posterior[t]
        Sigma_drkf = drkf_posterior[t]
        Sigma_high = high_posterior[t]
        
        # Check Loewner order and print margins if violated
        low_order, low_min_eig = check_loewner_order(Sigma_low, Sigma_drkf, f"LOW ⪯ DRKF t={t}")
        high_order, high_min_eig = check_loewner_order(Sigma_drkf, Sigma_high, f"DRKF ⪯ HIGH t={t}")
        
        if not (low_order and high_order):
            print(f"WARNING: Sandwich property violated at t={t}")
            print(f"  LOW ⪯ DRKF: {low_order} (min_eig: {low_min_eig:.6f})")
            print(f"  DRKF ⪯ HIGH: {high_order} (min_eig: {high_min_eig:.6f})")
        
        # Plot ellipses
        plot_ellipse(ax, Sigma_low, style_dict=low_style, label='LOW-KF' if i == 0 else None)
        plot_ellipse(ax, Sigma_drkf, style_dict=drkf_style, label='DRKF' if i == 0 else None)
        plot_ellipse(ax, Sigma_high, style_dict=high_style, label='HIGH-KF' if i == 0 else None)
        
        # Set equal aspect ratio and consistent limits across all plots
        ax.set_aspect('equal')
        ax.set_xlim(-global_limit, global_limit)
        ax.set_ylim(-global_limit, global_limit)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('$x_1$', fontsize=18)
        ax.set_ylabel('$x_2$', fontsize=18)
        
        # Add eigenvalue info as text
        eig_low = np.linalg.eigvals(Sigma_low)
        eig_drkf = np.linalg.eigvals(Sigma_drkf)
        eig_high = np.linalg.eigvals(Sigma_high)
        
        info_text = f'λ(LOW): [{eig_low[0]:.3f}, {eig_low[1]:.3f}]\n'
        info_text += f'λ(DRKF): [{eig_drkf[0]:.3f}, {eig_drkf[1]:.3f}]\n'
        info_text += f'λ(HIGH): [{eig_high[0]:.3f}, {eig_high[1]:.3f}]'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add overall title first
    # fig.suptitle(r'95% Confidence Ellipses: KF Sandwich Property' + '\n' + 
    #              r'$\theta_w=' + f'{theta_w}' + r'$, $\theta_v=' + f'{theta_v}' + r'$', 
    #              fontsize=16, y=0.96)
    
    # Add overall legend with box-shaped markers
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black', linestyle='--', linewidth=2, label='LOW-KF'),
                      Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', linestyle='-', linewidth=3, label='DRKF'),
                      Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', linestyle=':', linewidth=2, label='HIGH-KF')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=3, fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    # Save figure
    plt.savefig('ellipses_2D.pdf', bbox_inches='tight')
    print(f"Figure saved as 'ellipses_2D.pdf'")
    
    # Print summary statistics
    print("\nSummary of Ellipse Areas (relative to LOW-KF):")
    print("-" * 40)
    for i, t in enumerate(Tview):
        Sigma_low = low_posterior[t]
        Sigma_drkf = drkf_posterior[t]
        Sigma_high = high_posterior[t]
        
        # Ellipse area is proportional to sqrt(det(Sigma))
        area_low = np.sqrt(np.linalg.det(Sigma_low))
        area_drkf = np.sqrt(np.linalg.det(Sigma_drkf))
        area_high = np.sqrt(np.linalg.det(Sigma_high))
        
        ratio_drkf = area_drkf / area_low
        ratio_high = area_high / area_low
        
        print(f"t={t:2d}: DRKF/LOW = {ratio_drkf:.3f}, HIGH/LOW = {ratio_high:.3f}")
    
    # Check overall verification results
    spectral_pass = results['spectral_bounds_check']['all_bounds_satisfied']
    sandwich_pass = results['sandwich_check']['all_sandwich_satisfied']
    
    print(f"\nOverall Verification Results:")
    print(f"Spectral Bounds: {'PASS' if spectral_pass else 'FAIL'}")
    print(f"Sandwich Property: {'PASS' if sandwich_pass else 'FAIL'}")
    print(f"Overall: {'PASS' if spectral_pass and sandwich_pass else 'FAIL'}")
    
    # Generate 3D tube visualization
    print("\nGenerating 3D tube visualization...")
    fig_3d = plot_3d_tube(low_posterior, drkf_posterior, high_posterior, T)
    
    plt.show()

if __name__ == "__main__":
    main()