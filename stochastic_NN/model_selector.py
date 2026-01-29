import numpy as np
import copy
from scipy.stats import norm
from stochastic_NN import StochasticNN

class SNNStepwiseSelector:
    def __init__(self, max_lags=10, max_J=3):
        self.max_lags = max_lags
        self.max_J = max_J

    def get_bic(self, model, datasets):
        n_total = sum(len(d[1]) for d in datasets)
        log_lik = 0
        
        # Calculate log-likelihood based on Section 4.1
        for X_raw, y in datasets:
            X = model.get_X_subset(X_raw)
            y_hat = model.predict_expectation(X)
            log_lik += np.sum(norm.logpdf(y, loc=y_hat, scale=np.sqrt(model.sigma_sq)))
        
        # Exact parameter count (only non-zero parameters)
        m_linear = (1 if model.beta_0 != 0 else 0) + np.count_nonzero(model.b_0) + \
                   np.count_nonzero(model.beta) + np.count_nonzero(model.b)
        m_nonlinear = np.count_nonzero(model.alpha) + np.count_nonzero(model.a)
        m = m_linear + m_nonlinear + 1  # +1 for sigma_sq
        
        return -2 * log_lik + m * np.log(n_total)

    def run_selection(self, datasets):
        # Forward Selection
        selected_lags = []
        best_overall_bic = np.inf
        best_model = None
        for _ in range(self.max_lags):
            trial_bics = []
            remaining_lags = [i for i in range(1, self.max_lags + 1) if i not in selected_lags]
            
            if not remaining_lags:
                break
            
            for lag in remaining_lags:
                current_lags = selected_lags + [lag]
                model = StochasticNN(current_lags, J=1)
                model.train_on_multiple(datasets, iterations=15)
                trial_bics.append((self.get_bic(model, datasets), lag, model))
            
            trial_bics.sort()
            if trial_bics[0][0] < best_overall_bic:
                best_overall_bic = trial_bics[0][0]
                selected_lags.append(trial_bics[0][1])
                best_model = trial_bics[0][2]
                print(f"Added lag {trial_bics[0][1]}, BIC: {best_overall_bic:.2f}")
            else:
                print("No improvement, stopping forward selection.")
                break

        if best_model is None:
            # Fallback: use lag 1 if something went wrong
            selected_lags = [1]
            best_model = StochasticNN([1], J=1)
            best_model.train_on_multiple(datasets, iterations=15)
            best_overall_bic = self.get_bic(best_model, datasets)

        # Selecting J
        for j in range(2, self.max_J + 1):
            model = StochasticNN(selected_lags, J=j)
            model.train_on_multiple(datasets, iterations=20)
            current_bic = self.get_bic(model, datasets)
            if current_bic < best_overall_bic:
                best_overall_bic = current_bic
                best_model = model
                print(f"Increased J to {j}, New BIC: {best_overall_bic:.2f}")
            else:
                print(f"J={j} did not improve BIC, keeping J={best_model.J}")
                break
                
        # Backward Elimination 
        current_model = best_model
        current_bic = self.get_bic(current_model, datasets)
        improved = True
        elimination_round = 0
        while improved:
            improved = False
            best_elimination = None
            best_elimination_bic = current_bic
            elimination_round += 1
            candidates = []
            
            # 1. Test eliminating b_0[i] parameters
            for i in range(current_model.p):
                if not current_model.frozen_mask['b_0'][i] and current_model.b_0[i] != 0:
                    candidates.append(('b_0', None, i))
            
            # 2. Test eliminating b[j, i] parameters
            for j in range(current_model.J):
                for i in range(current_model.p):
                    if not current_model.frozen_mask['b'][j, i] and current_model.b[j, i] != 0:
                        candidates.append(('b', j, i))
            
            # 3. Test eliminating a[j, i] parameters
            for j in range(current_model.J):
                for i in range(current_model.p):
                    if not current_model.frozen_mask['a'][j, i] and current_model.a[j, i] != 0:
                        candidates.append(('a', j, i))
            
            if not candidates:
                print("No more parameters to eliminate.")
                break
            
            for param_type, j, i in candidates:
                # Create trial model with parameter frozen
                trial_model = copy.deepcopy(current_model)
                trial_model.freeze_parameter(param_type, j, i)
                
                # Re-train with frozen parameter (EM respects frozen mask)
                trial_model.train_on_multiple(datasets, iterations=5)
                
                trial_bic = self.get_bic(trial_model, datasets)
                
                if trial_bic < best_elimination_bic:
                    best_elimination_bic = trial_bic
                    best_elimination = (trial_model, param_type, j, i)
                    improved = True
            
            if improved:
                current_model = best_elimination[0]
                current_bic = best_elimination_bic
                param_type, j, i = best_elimination[1:]

        print(f"\nFinal model: {current_model.J} hidden units, {current_model.p_indices} lags")
        print(f"Final BIC: {current_bic:.2f}")
        
        return current_model