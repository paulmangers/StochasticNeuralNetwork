import numpy as np
from BS_delta_hedge import BSModel
from wealth_process import WealthProcess

# Example usage:
# Parameters
S0 = 110
K = 100
mu = 0.03
sigma = 0.25
r = 0.01
T = 1.0
N = 1000
dt = T / N
M = 1000
hedging_errors = []
avg_final_wealth = 0
avg_final_payoff = 0


model = BSModel(S0, mu, sigma, r, K, T, N)
for m in range(M):
    price_process = model.bs_sampler_underQ()
    delta_strategy = lambda i: model.bs_delta_hedge(price_process[i], tau=(T - i*dt), option_type='call')
    initial_wealth = model.bs_call_price(price_process[0], T)
    hedge = WealthProcess(initial_wealth, delta_strategy, price_process, model=model)
    wealth_path = hedge.simulate()
    final_wealth = wealth_path[-1]
    final_payoff = max(price_process[-1] - K, 0)
    hedging_error = abs(final_wealth - final_payoff)
    hedging_errors.append(hedging_error)
    avg_final_wealth += final_wealth / M
    avg_final_payoff += final_payoff / M

print(f"Average hedging error over {M} simulations: {np.mean(hedging_errors):.4f}") 
print(f"Standard deviation of hedging error over {M} simulations: {np.std(hedging_errors):.4f}")
print(f"Average final wealth over {M} simulations: {avg_final_wealth:.4f}")
print(f"Average final payoff over {M} simulations: {avg_final_payoff:.4f}")