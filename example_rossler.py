import numpy as np
import matplotlib.pyplot as plt
from lyapunov import DynamicalSystem, LyapunovSpectrum


# --------------------------
# Rossler System
# --------------------------

def rossler(t, state, a=0.15, b=0.2, c=10.0):
    x, y, z = state
    return np.array([
        -(y + z),
        x + a * y,
        b + z * (x - c)
    ])

def rossler_jacobian(state, a=0.15, b=0.2, c=10.0):
    x, y, z = state
    return np.array([
        [0.0, -1.0, -1.0],
        [1.0, a, 0.0],
        [z, 0.0, x - c]
    ])


# --------------------------
# Run Example
# --------------------------

if __name__ == "__main__":
    # setup system
    system = DynamicalSystem(f=rossler, jacobian=rossler_jacobian, dim=3)
    x0 = np.array([0.1, 0.0, 0.0])
    v0 = np.array([1.0, 1.0, 1.0])
    spectrum = LyapunovSpectrum(system, x0, v0)

    # compute exponents
    lyap_data, times = spectrum.compute(dt=0.01, T=400.0,
                                        reorthonormalize_every=10,
                                        log_base=np.e)

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
