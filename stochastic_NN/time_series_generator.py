import numpy as np

class TimeSeriesGenerator:
    def __init__(self, T, N):
        """
        T: Time horizon 
        N: Number of time steps 
        """
        self.T = T
        self.N = N
        self.dt = T / N

    def ito_process(self, mufunc, sigfunc, z0):
        z = np.zeros(self.N + 1)
        z[0] = z0
        for t in range(1, self.N + 1):
            t_prev = (t - 1) * self.dt
            dw = np.random.normal(0, np.sqrt(self.dt))
            z[t] = z[t - 1] + mufunc(t_prev, z[t - 1]) * self.dt + sigfunc(t_prev, z[t - 1]) * dw
        return z

    def generate_tar(self, z0, threshold=0.0, phi_high=0.9, phi_low=-0.5, sigma=0.1):
        """
        Threshold Autoregressive (TAR) Model.
        """
        z = np.zeros(self.N + 1)
        z[0] = z0
        for t in range(1, self.N + 1):
            eps = np.random.normal(0, sigma)
            # Switch regime based on the previous value
            if z[t-1] > threshold:
                z[t] = phi_high * z[t-1] + eps
            else:
                z[t] = phi_low * z[t-1] + eps
        return z

    def generate_logistic(self, z0, r=4.0, noise_std=0.0):
        """
        Logistic Map: z_{t+1} = r * z_t * (1 - z_t)
        Produces chaotic deterministic behavior when r=4.0.
        z0 must be between 0 and 1.
        """
        if not (0 <= z0 <= 1):
            raise ValueError("z0 for Logistic Map must be between 0 and 1.")
            
        z = np.zeros(self.N + 1)
        z[0] = z0
        for t in range(1, self.N + 1):
            # Chaotic core
            val = r * z[t-1] * (1 - z[t-1])
            # Add optional noise to test SNN's ability to filter randomness
            z[t] = np.clip(val + np.random.normal(0, noise_std), 0, 1)
        return z
        


