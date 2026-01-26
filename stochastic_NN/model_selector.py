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
                # Use the exact density based on the sigma^2 estimated by the model
                log_lik += np.sum(norm.logpdf(y, loc=y_hat, scale=np.sqrt(model.sigma_sq)))
            
            # Exact parameter count m (Section 4.2.3: Only count non-zero parameters)
            # This is crucial for backward elimination to work
            m_linear = np.count_nonzero(model.beta_0) + np.count_nonzero(model.b_0) + \
                    np.count_nonzero(model.beta) + np.count_nonzero(model.b)
            m_nonlinear = np.count_nonzero(model.alpha) + np.count_nonzero(model.a)
            m = m_linear + m_nonlinear + 1 # +1 for sigma_sq
            
            return -2 * log_lik + m * np.log(n_total)

    def run_selection(self, datasets):
       # --- Step 1: Forward Selection ---
        selected_lags = []
        best_overall_bic = np.inf
        
        print("Starting Forward Selection...")
        for _ in range(self.max_lags):
            trial_bics = []
            remaining_lags = [i for i in range(1, self.max_lags + 1) if i not in selected_lags]
            
            for lag in remaining_lags:
                current_lags = selected_lags + [lag]
                model = StochasticNN(current_lags, J=1)
                model.train_on_multiple(datasets, iterations=10) # Increased iterations
                trial_bics.append((self.get_bic(model, datasets), lag, model))
            
            trial_bics.sort()
            if trial_bics[0][0] < best_overall_bic:
                best_overall_bic = trial_bics[0][0]
                selected_lags.append(trial_bics[0][1])
                # IMPORTANT: Keep track of the actual model object
                best_model = trial_bics[0][2] 
                print(f"Added lag {trial_bics[0][1]}, BIC: {best_overall_bic:.2f}")
            else:
                break

        # --- Step 2: Selection of J ---
        print(f"Selecting J for lags {selected_lags}...")
        # We already have a best_model for J=1 from Step 1. 
        # We only check if J > 1 improves it.
        for j in range(2, self.max_J + 1):
            model = StochasticNN(selected_lags, J=j)
            model.train_on_multiple(datasets, iterations=20)
            current_bic = self.get_bic(model, datasets)
            if current_bic < best_overall_bic:
                best_overall_bic = current_bic
                best_model = model
                print(f"Increased J to {j}, New BIC: {best_overall_bic:.2f}")
                
        # Step 3: Backward Elimination (Pruning) as per Section 4.2.3
        print("Finalizing via Backward Elimination (Exact Implementation)...")
        current_model = best_model
        current_bic = self.get_bic(current_model, datasets)
        
        improved = True
        while improved:
            improved = False
            best_pruned_model = None
            
            # 1. Identify all currently non-zero parameters
            # We check coefficients in b_0, b[j], and a[j]
            # (The constants beta and alpha are usually kept unless specified)
            
            # Test pruning each lag coefficient in the global linear part
            for i in range(current_model.p):
                if current_model.b_0[i] == 0: continue
                
                trial_model = copy.deepcopy(current_model)
                trial_model.b_0[i] = 0
                # Re-train to see if other params compensate (Section 4.2.3 requirement)
                # Note: StochasticNN.train_on_multiple must respect zeros (no updates to 0s)
                trial_model.train_on_multiple(datasets, iterations=5) 
                
                trial_bic = self.get_bic(trial_model, datasets)
                if trial_bic < current_bic:
                    current_bic = trial_bic
                    best_pruned_model = trial_model
                    improved = True

            # Test pruning each lag coefficient in each hidden unit j
            for j in range(current_model.J):
                for i in range(current_model.p):
                    # Check linear coefficients b[j, i]
                    if current_model.b[j, i] != 0:
                        trial_model = copy.deepcopy(current_model)
                        trial_model.b[j, i] = 0
                        trial_model.train_on_multiple(datasets, iterations=5)
                        trial_bic = self.get_bic(trial_model, datasets)
                        if trial_bic < current_bic:
                            current_bic = trial_bic
                            best_pruned_model = trial_model
                            improved = True
                    
                    # Check nonlinear gating coefficients a[j, i]
                    if current_model.a[j, i] != 0:
                        trial_model = copy.deepcopy(current_model)
                        trial_model.a[j, i] = 0
                        trial_model.train_on_multiple(datasets, iterations=5)
                        trial_bic = self.get_bic(trial_model, datasets)
                        if trial_bic < current_bic:
                            current_bic = trial_bic
                            best_pruned_model = trial_model
                            improved = True
            
            if improved:
                current_model = best_pruned_model
                print(f"Pruned a parameter. New BIC: {current_bic:.2f}")

        return current_model