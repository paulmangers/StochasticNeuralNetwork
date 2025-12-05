import numpy as np
import matplotlib.pyplot as plt

class WealthProcess:
    def __init__(self, initial_wealth, strategy, price_process, model):
        self.initial_wealth = initial_wealth
        self.strategy = strategy
        self.price_process = price_process 
        self.model = model
        self.path = None

    def simulate(self):
        N = self.model.N
        self.path = np.zeros(N+1)
        self.path[0] = self.initial_wealth

        for i in range(1, N+1):
            prev_wealth = self.path[i-1]
            phi = self.strategy(i-1)  
            dS = self.price_process[i] - self.price_process[i-1]
            cash_portion = prev_wealth - phi * self.price_process[i-1]
            cash_curr = cash_portion * np.exp(self.model.r * self.model.dt) 
            self.path[i] = prev_wealth + phi * dS + cash_curr - cash_portion

        return self.path

    def plot_path(self):
        times = np.linspace(0, self.model.T, len(self.path))
        plt.plot(times, self.path)
        plt.xlabel("Time")
        plt.ylabel("Wealth")
        plt.title("Delta Hedge Wealth Process")
        plt.show()
