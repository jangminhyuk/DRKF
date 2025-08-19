On The Steady-State Distributionally Robust Kalman Filter
====================================================

This repository contains code to run experiments on a steady–state, distributionally robust Kalman filter. 
The experiments compare the performance of different filtering methods under various uncertainty distributions.

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)** (required by CVXPY for solving optimization problems)
- (pickle5) install if you encounter compatibility issues with pickle
- joblib (>=1.4.2, Used for parallel computation)

### main.py

Runs closed–loop simulations applying an LQR controller with various filters. The experiments consider both Gaussian and U–Quadratic uncertainty distributions. After the experiments, results (including optimal robust parameter selection) are saved.

- For Gaussian uncertainties

```
python main.py --dist normal
```

- For U-Quadratic uncertainties

```
python main.py --dist quadratic
```

- For Laplace uncertainties

```
python main.py --dist laplace
```

### convergence_check.py

```
python convergence_check.py
```

### main2.py

Closed-loop simulation with LQR controller and trajectory tracking using 7 different filters. Compares robust filters' performance under various uncertainty distributions. Finds optimal robust parameters for each filter based on LQR cost.

```
python main2.py --dist normal
```

### main3.py

Open-loop estimation experiments without control to test pure estimator performance. Runs the same 7 filters in open-loop mode with zero control input. Evaluates MSE performance and finds optimal robust parameters.

```
python main3.py --dist normal
```

### drkf_tube_check.py

Verifies DRKF spectral boundedness and KF sandwich properties. Runs 20 random experiments with different systems and parameters. Reports success rates for theoretical bound verification.

```
python drkf_tube_check.py
```