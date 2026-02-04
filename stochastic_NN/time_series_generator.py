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

    def generate_tar(self, z0, threshold=0.0, phi_high=0.7, phi_low=-0.5, sigma=0.1):
        """
        Threshold Autoregressive (TAR) Model.
        """
        z = np.zeros(self.N + 1)
        z[0] = z0
        for t in range(1, self.N + 1):
            eps = np.random.normal(0, sigma)
            if z[t-1] > threshold:
                z[t] = phi_high * z[t-1] + eps
            else:
                z[t] = phi_low * z[t-1] + eps
        return z
    
    def generate_bull_bear_sideways(self):
        """
        3-regime model inspired by market states.
        
        Bull market: positive drift, low volatility
        Bear market: negative drift, high volatility  
        Sideways: no drift, moderate volatility
        """
        z = np.zeros(self.N + 1)
        z[0] = 0.0
        regime_state = np.zeros(self.N + 1, dtype=int)
        regime_state[0] = 1  # Start in sideways
        
        for t in range(1, self.N + 1):
            
            if regime_state[t-1] == 0:  # Bull
                drift = 0.15
                phi = 0.75
                sigma = 0.1
            elif regime_state[t-1] == 1:  # Sideways
                drift = 0.0
                phi = 0.4
                sigma = 0.20
            else:  # regime_state[t-1] == 2, Bear
                drift = -0.25
                phi = 0.7
                sigma = 0.4
            
            eps = np.random.normal(0, sigma)
            z[t] = drift + phi*z[t-1] + eps
            
            # Regime transitions (endogenous)
            if z[t] > 0.8:
                regime_state[t] = 0  # Enter bull when high
            elif z[t] < -0.8:
                regime_state[t] = 2  # Enter bear when low
            else:
                # Stay in sideways or transition back
                if regime_state[t-1] in [0, 2]:
                    # With 20% probability, return to sideways
                    if np.random.rand() < 0.2:
                        regime_state[t] = 1
                    else:
                        regime_state[t] = regime_state[t-1]
                else:
                    regime_state[t] = 1
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
        

    # ========================================================================
    # EXAMPLES FROM PAPER
    # ========================================================================
    
    def generate_example_6_1(self, n=100, burn_in=1000):
        """
        Example 6.1: Piecewise Linear Autoregression (Equation 17)

        Parameters:
            n: Number of observations to return
            burn_in: Number of initial observations to discard for stationarity
        
        Returns:
            Array of length n+3 (need y_{-2}, y_{-1}, y_0 for initial conditions)
        """
        total_n = burn_in + n + 3
        y = np.zeros(total_n)
        
        # Initialize with small random values
        y[0] = np.random.normal(0, 0.1)
        y[1] = np.random.normal(0, 0.1)
        y[2] = np.random.normal(0, 0.1)
        
        for t in range(3, total_n):
            eps = np.random.normal(0, 1)

            if 2*y[t-2] < y[t-1] + y[t-3]:
                # Regime 1
                y[t] = 1 + 0.7*y[t-1] + 0.05*y[t-2] + eps
            else:
                # Regime 2
                y[t] = 0.8*y[t-1] + eps

        return y[burn_in:]
    
    def generate_example_6_2(self, n=300, burn_in=1000):
        """
        Example 6.2: Markov Chain (Equation 19)
    
        Parameters:
            n: Number of observations to return
            burn_in: Number of initial observations to discard
        
        Returns:
            Array of length n (values in [0, 1])
        """
        total_n = burn_in + n
        y = np.zeros(total_n)
        
        # Initialize from stationary distribution π(y) = 2(1-y)
        # This is Beta(1, 2) distribution
        y[0] = np.random.beta(1, 2)
        
        for t in range(1, total_n):
            x = y[t-1]
            
            # Generate from transition density p(y|x) using inverse transform sampling
            u = np.random.uniform(0, 1)
            
            # Inverse CDF
            if u <= x * np.exp(1-x):
                y[t] = u / np.exp(1-x)
            else:
                max_density = np.exp(1-x) - 1 
                while True:
                    y_candidate = np.random.uniform(x, 1)
                    density = np.exp(1-x) - np.exp(y_candidate - x)
                    
                    if np.random.uniform(0, max_density) <= density:
                        y[t] = y_candidate
                        break
        
        return y[burn_in:]
    



