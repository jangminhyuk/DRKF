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
    
    # Time steps to sample for tube construction (starting from t=0)
    # Use every single time step to show smooth evolution
    time_steps = np.arange(0, T + 1, 1)  # Sample every time step
    
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
    
    # Create smooth tube surfaces by sampling more time steps
    tube_times = np.arange(1, T + 1, 1)  # Every time step from t=1 to T
    
    # Generate tube surfaces for each filter
    all_low_t, all_low_x, all_low_y = [], [], []
    all_drkf_t, all_drkf_x, all_drkf_y = [], [], []
    all_high_t, all_high_x, all_high_y = [], [], []
    
    for t in tube_times:
        # Get ellipse points for each filter at this time step
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
    low_t_surf = np.array(all_low_t)
    low_x_surf = np.array(all_low_x)
    low_y_surf = np.array(all_low_y)
    
    drkf_t_surf = np.array(all_drkf_t)
    drkf_x_surf = np.array(all_drkf_x)
    drkf_y_surf = np.array(all_drkf_y)
    
    high_t_surf = np.array(all_high_t)
    high_x_surf = np.array(all_high_x)
    high_y_surf = np.array(all_high_y)
    
    # Plot nested tube surfaces with careful alpha settings for visibility
    # HIGH-KF (outermost) - Very transparent red
    ax.plot_surface(high_t_surf, high_x_surf, high_y_surf, 
                    alpha=0.15, color='red', label='HIGH-KF')
    
    # DRKF (middle) - Semi-transparent green  
    ax.plot_surface(drkf_t_surf, drkf_x_surf, drkf_y_surf, 
                    alpha=0.35, color='green', label='DRKF')
    
    # LOW-KF (innermost) - More opaque blue for visibility
    ax.plot_surface(low_t_surf, low_x_surf, low_y_surf, 
                    alpha=0.55, color='blue', label='LOW-KF')
    
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
    
    # Set time axis to show only integer values
    ax.set_xticks(range(0, T + 1, 5))  # Every 5 time steps
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    
    # Save figure with extra padding to prevent cropping
    plt.tight_layout()
    plt.savefig('tube_3D.pdf', bbox_inches='tight', pad_inches=0.5)
    print(f"3D tube figure saved as 'tube_3D.pdf'")
    
    return fig

def create_time_varying_covariances(nominal_Sigma_w, nominal_Sigma_v, T, method='exponential_decay'):
    """
    Create time-varying covariance trajectories using different methods.
    
    Parameters:
    -----------
    nominal_Sigma_w : ndarray
        Base process noise covariance
    nominal_Sigma_v : ndarray
        Base measurement noise covariance
    T : int
        Time horizon
    method : str
        Method for time variation: 'exponential_decay', 'oscillating_decay', 
        'step_function', 'sigmoid', 'piecewise_linear'
    
    Returns:
    --------
    tuple
        (Sigma_w_traj, Sigma_v_traj) - lists of covariance matrices
    """
    Sigma_w_traj = []
    Sigma_v_traj = []
    
    for t in range(T + 1):
        if method == 'exponential_decay':
            # Current method: exponential decay
            decay_factor = np.exp(-0.1 * t)
            w_factor = 0.3 + 0.7 * decay_factor
            v_factor = 0.5 + 0.5 * decay_factor
            
        elif method == 'oscillating_decay':
            # Oscillating with decay - creates twisted funnel
            osc_factor = 1 + 0.5 * np.sin(2*np.pi*t/8)  # 8-step period
            decay_factor = np.exp(-0.05 * t)
            w_factor = osc_factor * decay_factor
            v_factor = 0.5 + 0.3 * osc_factor * decay_factor
            
        elif method == 'step_function':
            # Step decreases every 5 time steps
            w_factor = 1.0 / (1 + t // 5)
            v_factor = 0.8 / (1 + t // 6)
            
        elif method == 'sigmoid':
            # Sigmoid transition from 1.0 to 0.2
            x = 8 * (t - T/2) / T  # Scale to [-4, 4]
            w_factor = 0.2 + 0.8 / (1 + np.exp(x))
            v_factor = 0.3 + 0.5 / (1 + np.exp(x))
            
        elif method == 'piecewise_linear':
            # Three-phase evolution
            if t <= 5:
                w_factor = 1.0 - 0.6 * t/5  # Fast initial drop
                v_factor = 1.0 - 0.4 * t/5
            elif t <= 15:
                w_factor = 0.4 - 0.2 * (t-5)/10  # Slow middle
                v_factor = 0.6 - 0.3 * (t-5)/10
            else:
                w_factor = 0.2  # Constant end
                v_factor = 0.3
                
        else:
            raise ValueError(f"Unknown method: {method}")
        
        Sigma_w_t = nominal_Sigma_w * max(w_factor, 0.1)  # Minimum factor
        Sigma_v_t = nominal_Sigma_v * max(v_factor, 0.1)
        
        Sigma_w_traj.append(Sigma_w_t)
        Sigma_v_traj.append(Sigma_v_t)
    
    return Sigma_w_traj, Sigma_v_traj

def main():
    print("2D Ellipse Visualization of KF Sandwich Property")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define 2D system
    nx, ny = 2, 2
    T = 20
    
    # Create oscillatory system with complex eigenvalues for interesting evolution
    # This creates a rotation with slight decay (eigenvalues ~0.95*exp(±iθ))
    theta = np.pi/8  # 22.5 degrees rotation per step
    decay = 0.95     # Slight decay to make filter work harder
    A = decay * np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta), np.cos(theta)]])
    C = np.array([[1.0, 0.1], 
                  [0.05, 1.0]])
    
    print(f"System dimensions: nx={nx}, ny={ny}")
    print(f"Time horizon: T={T}")
    print(f"A matrix eigenvalues: {np.linalg.eigvals(A)}")
    
    # Generate time-varying covariances for evolving tube shape
    # Start with large process noise that decreases over time
    nominal_Sigma_w = np.array([[0.30, 0.10], 
                                [0.10, 0.25]])
    # Measurement noise also varies
    nominal_Sigma_v = np.array([[0.15, 0.05], 
                                [0.05, 0.12]])
    # Very large initial uncertainty - this creates the funnel effect
    nominal_x0_posterior = np.array([[0.2, 0.06], 
                                     [0.06, 0.16]])
    
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
    
    # Large Wasserstein radii for visible separation
    theta_w = 0.15
    theta_v = 0.10
    
    print(f"Wasserstein radii: θ_w={theta_w}, θ_v={theta_v}")
    print()
    
    # Test different time-varying covariance methods
    methods = ['exponential_decay', 'oscillating_decay', 'step_function', 'sigmoid', 'piecewise_linear']
    
    print("\nTesting different time-varying covariance methods:")
    print("=" * 60)
    
    for method in methods:
        print(f"\n--- Method: {method.upper().replace('_', ' ')} ---")
        
        # Create time-varying trajectories
        nominal_Sigma_w_traj, nominal_Sigma_v_traj = create_time_varying_covariances(
            nominal_Sigma_w, nominal_Sigma_v, T, method=method
        )
    
        # Run verification
        print(f"Running DRKF verification for {method}...")
        results = drkf_spectral_verification(
            A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj,
            theta_w, theta_v, nominal_x0_posterior, T,
            verbose=False
        )
        
        # Extract covariance trajectories
        low_posterior = results['low_kf']['Sigma_posterior']
        high_posterior = results['high_kf']['Sigma_posterior']
        drkf_posterior = results['drkf']['Sigma_x_posterior']
        
        print(f"Verification completed for {method}!")
        
        # Print evolution statistics
        initial_area = np.sqrt(np.linalg.det(drkf_posterior[0]))
        final_area = np.sqrt(np.linalg.det(drkf_posterior[T]))
        print(f"DRKF area evolution: {initial_area:.4f} -> {final_area:.4f} (ratio: {final_area/initial_area:.3f})")
    
    # Time points to visualize
    Tview = [0, 1, 5, 20]
    
    # Style definitions
    low_style = {'color': 'blue', 'linestyle': '--', 'linewidth': 2}
    drkf_style = {'color': 'green', 'linestyle': '-', 'linewidth': 3}
    high_style = {'color': 'red', 'linestyle': ':', 'linewidth': 2}
    
    for method in methods:
        print(f"\n--- Method: {method.upper().replace('_', ' ')} ---")
        
        # Create time-varying trajectories
        nominal_Sigma_w_traj, nominal_Sigma_v_traj = create_time_varying_covariances(
            nominal_Sigma_w, nominal_Sigma_v, T, method=method
        )
        
        # Run verification
        print(f"Running DRKF verification for {method}...")
        results = drkf_spectral_verification(
            A, C, nominal_Sigma_w_traj, nominal_Sigma_v_traj,
            theta_w, theta_v, nominal_x0_posterior, T,
            verbose=False
        )
        
        # Extract covariance trajectories
        low_posterior = results['low_kf']['Sigma_posterior']
        high_posterior = results['high_kf']['Sigma_posterior']
        drkf_posterior = results['drkf']['Sigma_x_posterior']
        
        print(f"Verification completed for {method}!")
        
        # Print evolution statistics
        initial_area = np.sqrt(np.linalg.det(drkf_posterior[0]))
        final_area = np.sqrt(np.linalg.det(drkf_posterior[T]))
        print(f"DRKF area evolution: {initial_area:.4f} -> {final_area:.4f} (ratio: {final_area/initial_area:.3f})")
        
        # Generate 2D visualization for this method
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
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
            
            # Plot ellipses
            plot_ellipse(ax, Sigma_high, style_dict=high_style, label='HIGH-KF' if i == 0 else None)
            plot_ellipse(ax, Sigma_low, style_dict=low_style, label='LOW-KF' if i == 0 else None)
            plot_ellipse(ax, Sigma_drkf, style_dict=drkf_style, label='DRKF' if i == 0 else None)
            
            # Set equal aspect ratio and consistent limits across all plots
            ax.set_aspect('equal')
            ax.set_xlim(-global_limit, global_limit)
            ax.set_ylim(-global_limit, global_limit)
            
            # Add grid and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('$x_1$', fontsize=18)
            ax.set_ylabel('$x_2$', fontsize=18)
            ax.set_title(f't = {t}', fontsize=20)
        
        # Add legend and save
        from matplotlib.patches import Rectangle
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black', linestyle='--', linewidth=2, label='LOW-KF'),
                          Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', linestyle='-', linewidth=3, label='DRKF'),
                          Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', linestyle=':', linewidth=2, label='HIGH-KF')]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=18)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save with method name
        filename = f'ellipses_2D_{method}.pdf'
        plt.savefig(filename, bbox_inches='tight')
        print(f"2D figure saved as '{filename}'")
        
        # Generate 3D tube visualization for this method
        print(f"Generating 3D tube for {method}...")
        fig_3d = plot_3d_tube(low_posterior, drkf_posterior, high_posterior, T)
        
        # Save 3D plot with method name
        filename_3d = f'tube_3D_{method}.pdf'
        plt.savefig(filename_3d, bbox_inches='tight', pad_inches=0.5)
        print(f"3D figure saved as '{filename_3d}'")
        
        plt.close('all')  # Close figures to save memory
    
    print("\nAll methods completed! Check the generated PDF files.")

if __name__ == "__main__":
    main()