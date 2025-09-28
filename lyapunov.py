import numpy as np
from scipy.integrate import solve_ivp


# --------------------------
# Dynamical System
# --------------------------

class DynamicalSystem:
    def __init__(self, f, jacobian, dim):
        """
        f        : function f(t, x) returning dx/dt
        jacobian : function J(x) returning Jacobian matrix
        dim      : dimension of the state space
        """
        self.f = f
        self.jacobian = jacobian
        self.dim = dim

    def combined_dynamics(self, t, full_state):
        """
        Combined evolution of state + tangent vectors.
        full_state = [x, V.flatten()]
        """
        x = full_state[:self.dim]
        V = full_state[self.dim:].reshape(self.dim, self.dim)

        dx = self.f(t, x)
        J = self.jacobian(x)
        dV = J @ V

        return np.concatenate([dx, dV.flatten()])

# --------------------------
# Utilities
# --------------------------

def gram_schmidt(V):
    """Orthonormalize columns of V using Gram-Schmidt process."""
    dim = V.shape[0]
    Q = np.zeros_like(V)
    norms = np.zeros(dim)

    for i in range(dim):
        v = V[:, i].copy()
        for j in range(i):
            v -= np.dot(Q[:, j], v) * Q[:, j]
        norms[i] = np.linalg.norm(v)
        Q[:, i] = v / (norms[i] + 1e-16)

    return norms, Q


def integrate_step(system, full_state, t0, dt,
                   method='RK45', rtol=1e-9, atol=1e-9):
    """Integrate system for one step of duration dt."""
    sol = solve_ivp(
        fun=lambda t, y: system.combined_dynamics(t, y),
        t_span=(t0, t0 + dt),
        y0=full_state,
        method=method,
        rtol=rtol,
        atol=atol
    )
    return sol.y[:, -1]

# --------------------------
# Lyapunov Spectrum
# --------------------------

class LyapunovSpectrum:
    def __init__(self, system, x0, v0):
        """
        system : DynamicalSystem instance
        x0     : initial condition for the system state
        v0     : initial tangent vectors
        """
        self.system = system
        self.x0 = np.array(x0, dtype=float)
        self.v0 = np.diag(np.array(v0, dtype=float))

    def compute(self, dt=0.01, T=500.0, transient=0, 
                reorthonormalize_every=10, log_base=np.e):
        """
        Compute Lyapunov exponents using the standard algorithm.
        
        Parameters
        ----------
        dt : float
            Integration timestep.

        T : float
            Total integration time.

        transient : float
            Transient time before accumulation starts.

        reorthonormalize_every : int
            Number of steps between reorthonormalizations.

        log_base : float
            Base of the logarithm for exponent calculation.
            np.log(2) for bits/iterations

        Returns
        -------
        history : ndarray
            Time series of Lyapunov exponents (shape: [n_steps, dim]).
        times : ndarray
            Array of times corresponding to `history`.
        """

        dim = self.system.dim
        steps = int(T / dt)
        step_size = dt * reorthonormalize_every

        # determine transient time
        transient_steps = int(transient / dt)

        # initial combined state
        full_state = np.concatenate([self.x0, self.v0.flatten()])
        cumulative_logs = np.zeros(dim)
        current_time = 0.0
        history = []

        for step in range(0, steps, reorthonormalize_every):
            # integrate system
            full_state = integrate_step(self.system, full_state,
                                        current_time, step_size)
            V = full_state[dim:].reshape(dim, dim)

            # reorthonormalize
            norms, Q = gram_schmidt(V)

            current_time += step_size

            # accumulate only after transient
            if step >= transient_steps:
                cumulative_logs += np.log(norms + 1e-16) / np.log(log_base)                
                history.append(cumulative_logs / current_time)

            # reset tangent vectors
            full_state = np.concatenate([full_state[:dim], Q.flatten()])          

        times = np.linspace(transient, T, len(history))
        return np.array(history), times
    
    