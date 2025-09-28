# lyapunov-ode

A minimal Python library for estimating the full Lyapunov spectrum of ordinary differential equation (ODE) systems. It uses tangent-space evolution with Gram–Schmidt reorthonormalization.

---

## Overview

`lyapunov-ode` provides an implementation of the standard algorithm for computing the **Lyapunov spectrum** of continuous-time systems.  
Given any user-defined ODE and its Jacobian, the library integrates both the system and its tangent space using `scipy.solve_ivp`.  
It periodically applies **Gram–Schmidt reorthonormalization** to maintain numerical stability and estimate all Lyapunov exponents.

This implementation emphasizes **clarity, reproducibility, and scientific correctness**, making it ideal for research, education, and exploration of chaotic systems.

---

## Features

- Compute **full Lyapunov spectrum** for ODE systems  
- Works with any user-defined system and its **Jacobian**
- Includes **transient period handling**
- Simple integration using **SciPy’s `solve_ivp`**
- Minimal dependencies (`numpy`, `scipy`, `matplotlib`)
- Example scripts for **Lorenz** and **Rössler** attractors

---

## Requirements

- Numpy
- SciPy
- Matplotlib (for examples and plotting)

---

## References

A. Wolf, J. B. Swift, H. L. Swinney, J. A. Vastano,
Determining Lyapunov exponents from a time series,
Physica D, 16(3):285–317, 1985.

---

## Author

Developed by Dr. Kağan Başkan.