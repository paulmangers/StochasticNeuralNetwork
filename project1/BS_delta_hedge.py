import numpy as np
from scipy.stats import norm

class BSModel:
    def __init__(self, S0, mu, sigma, r, K, T, N):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.K = K
        self.T = T
        self.N = N
        self.dt = T / N

    def bs_sampler_underQ(self):
        dW = np.random.normal(0, np.sqrt(self.dt), size=self.N)
        W = np.insert(np.cumsum(dW), 0, 0.0)
        t = np.linspace(0, self.T, self.N+1)
        S = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*t + self.sigma*W)
        return S  

    def bs_delta_hedge(self, S, tau, option_type='call'):
        if tau <= 0:
            return 1.0 if option_type == 'call' and S > self.K else 0.0 \
                   if option_type == 'call' else -1.0 if S < self.K else 0.0
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1

    def bs_call_price(self, S, tau):
        if tau <= 0:
            return max(S - self.K, 0.0)
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)
        return S * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
