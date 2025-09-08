#!/usr/bin/env python3
"""
Test script to verify reproducibility of main5.py
"""
import numpy as np
import sys
import os

# Add the current directory to path to import main5
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main5 functions
from main5 import main

def test_reproducibility():
    """Test if main5.py produces identical results across runs"""
    print("Testing reproducibility of main5.py...")
    
    # Use small parameters for quick test
    dist = "normal"
    num_sim = 1  # Small number of simulations
    num_exp = 2  # Small number of experiments
    T_total = 5  # Short time horizon
    
    print(f"Running test with: dist={dist}, num_sim={num_sim}, num_exp={num_exp}, T_total={T_total}")
    
    # Run the same experiment twice
    print("\n--- First run ---")
    results1 = main(dist, num_sim, num_exp, T_total)
    
    print("\n--- Second run ---")  
    results2 = main(dist, num_sim, num_exp, T_total)
    
    # Compare results
    print("\n--- Comparing results ---")
    identical = True
    
    # Check if same robust values were tested
    robust_vals1 = sorted(results1.keys())
    robust_vals2 = sorted(results2.keys())
    
    if robust_vals1 != robust_vals2:
        print(f"ERROR: Different robust values tested!")
        print(f"Run 1: {robust_vals1}")
        print(f"Run 2: {robust_vals2}")
        identical = False
    else:
        print(f"✓ Same robust values tested: {len(robust_vals1)} values")
    
    # Check MSE results for each robust value
    for rv in robust_vals1:
        if rv in results2:
            mse1 = results1[rv]['mse']
            mse2 = results2[rv]['mse']
            
            # Check if all filters have identical MSE values
            for filter_name in mse1:
                if filter_name in mse2:
                    if not np.allclose(mse1[filter_name], mse2[filter_name], atol=1e-10):
                        print(f"ERROR: MSE differs for {filter_name} at θ={rv}")
                        print(f"  Run 1: {mse1[filter_name]}")
                        print(f"  Run 2: {mse2[filter_name]}")
                        identical = False
                else:
                    print(f"ERROR: Filter {filter_name} missing in run 2 at θ={rv}")
                    identical = False
    
    if identical:
        print("\n✅ SUCCESS: Results are identical across runs!")
        print("main5.py is now reproducible.")
    else:
        print("\n❌ FAILURE: Results differ between runs!")
        print("main5.py is not reproducible.")
    
    return identical

if __name__ == "__main__":
    test_reproducibility()