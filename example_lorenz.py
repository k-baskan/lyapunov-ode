import numpy as np
import matplotlib.pyplot as plt
from lyapunov import DynamicalSystem, LyapunovSpectrum


# --------------------------
# Lorenz System
# --------------------------

def lorenz(t, state, sigma=16.0, rho=45.92, beta=4.0):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        -x * z + rho * x - y,
        x * y - beta * z
    ])

def lorenz_jacobian(state, sigma=16.0, rho=45.92, beta=4.0):
    x, y, z = state
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])


# --------------------------
# Run Example
# --------------------------

if __name__ == "__main__":
    # setup system
    system = DynamicalSystem(f=lorenz, jacobian=lorenz_jacobian, dim=3)
    x0 = np.array([10.0, 1.0, 0.0])
    v0 = np.array([1.0, 1.0, 1.0])
    spectrum = LyapunovSpectrum(system, x0, v0)

    # compute exponents
    lyap_data, times = spectrum.compute(dt=0.01, T=500.0, transient=5,
                                        reorthonormalize_every=10,
                                        log_base=2)

    # print final result
    print("Final Lyapunov Exponents:", lyap_data[-1])

    # plot
    plt.figure(figsize=(10, 5))
    for i in range(lyap_data.shape[1]):
        plt.plot(times, lyap_data[:, i], label=f"λ{i+1}")


    # Mark transient region
    if times[0] > 0:
        plt.axvspan(0, times[0], facecolor='none', edgecolor='gray', hatch='////', alpha=0.3, label='Transient')

    plt.xlabel("Time")
    plt.ylabel("Lyapunov Exponents")
    plt.title("Lyapunov Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
