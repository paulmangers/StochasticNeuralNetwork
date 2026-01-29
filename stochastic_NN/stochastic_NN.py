import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import itertools

class StochasticNN:
    def __init__(self, p_indices, J):
        """
        p_indices: List of specific lag indices to use (e.g., [1, 2, 5])
        J: Number of hidden units
        """
        self.p_indices = p_indices
        self.p = len(p_indices)
        self.J = J
        
        # Initialize with small random values to break symmetry
        self.beta_0 = 0.0
        self.b_0 = np.random.normal(0, 0.01, self.p)
        
        self.beta = np.random.normal(0, 0.01, J)
        self.b = np.random.normal(0, 0.01, (J, self.p))
        
        self.alpha = np.random.normal(0, 0.01, J)
        self.a = np.random.normal(0, 0.01, (J, self.p))
        
        self.sigma_sq = 1.0
        
        # Masks to track which parameters are frozen (for backward elimination)
        self.frozen_mask = {
            'b_0': np.zeros(self.p, dtype=bool),
            'b': np.zeros((J, self.p), dtype=bool),
            'a': np.zeros((J, self.p), dtype=bool)
        }

    def logistic(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def get_X_subset(self, X_full):
        return X_full[:, [i-1 for i in self.p_indices]]

    def predict_expectation(self, X):
        """
        Calculates E[y_t | x_t] as defined in Equation (12)
        """
        y_hat = self.beta_0 + np.dot(X, self.b_0)
        probs = self.logistic(self.alpha + np.dot(X, self.a.T))
        for j in range(self.J):
            y_hat += (self.beta[j] + np.dot(X, self.b[j])) * probs[:, j]
        return y_hat

    def _update_linear_parameters(self, datasets, all_weights):
        """
        Standard weighted least squares as per Section 4.1, Equation (16).

        """
        X_total = np.vstack([self.get_X_subset(d[0]) for d in datasets])
        y_total = np.concatenate([d[1] for d in datasets])
        W_total = np.vstack(all_weights)
        N = len(y_total)
        
        # Build design matrix
        X_design_parts = [np.ones((N, 1)), X_total]  
        
        for j in range(self.J):
            w_j = W_total[:, j:j+1]  # Shape (N, 1)
            X_design_parts.append(w_j)  
            X_design_parts.append(w_j * X_total)  
        
        X_design = np.hstack(X_design_parts)
        
        # Build parameter mask respecting frozen (pruned) parameters
        active_mask = [True]  # beta_0 is always active
        active_mask.extend(~self.frozen_mask['b_0'])
        
        for j in range(self.J):
            active_mask.append(True)  # beta_j is always active
            active_mask.extend(~self.frozen_mask['b'][j])
        
        active_mask = np.array(active_mask)
        active_indices = np.where(active_mask)[0]
        
        # Solve reduced system with L2 regularization for numerical stability
        X_active = X_design[:, active_indices]
        
        # Adaptive ridge penalty 
        ridge_lambda = 1e-6 * np.trace(X_active.T @ X_active) / len(active_indices)
        
        try:
            regularized_gram = X_active.T @ X_active + ridge_lambda * np.eye(len(active_indices))
            theta_reduced = np.linalg.solve(regularized_gram, X_active.T @ y_total)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if still singular
            theta_reduced = np.linalg.lstsq(X_active.T @ X_active, X_active.T @ y_total, rcond=None)[0]
        
        # Reconstruct full parameter vector
        theta_full = np.zeros(len(active_mask))
        theta_full[active_indices] = theta_reduced
        
        # Distribute back to model parameters
        idx = 0
        self.beta_0 = theta_full[idx]
        idx += 1
        
        self.b_0 = theta_full[idx:idx+self.p].copy()
        self.b_0[self.frozen_mask['b_0']] = 0  # Ensure frozen params stay zero
        idx += self.p
        
        for j in range(self.J):
            self.beta[j] = theta_full[idx]
            idx += 1
            self.b[j] = theta_full[idx:idx+self.p].copy()
            self.b[j][self.frozen_mask['b'][j]] = 0
            idx += self.p

    def _update_sigma_squared(self, datasets, all_weights, all_config_probs):
        """
        Conditional expectation to update sigma as per Section 4.1, Equation (16).
        
        """
        total_sq = 0
        n_total = 0
        
        configs = list(itertools.product([0, 1], repeat=self.J))
        
        for dataset_idx, (X_raw, y) in enumerate(datasets):
            X = self.get_X_subset(X_raw)
            N = X.shape[0]
            config_probs = all_config_probs[dataset_idx]  # Shape (N, 2^J)
            
            for idx, config in enumerate(configs):
                # Compute y_pred for this configuration
                y_pred = self.beta_0 + np.dot(X, self.b_0)
                for j in range(self.J):
                    if config[j] == 1:
                        y_pred += (self.beta[j] + np.dot(X, self.b[j]))
                
                residuals_sq = (y - y_pred) ** 2
                total_sq += np.sum(config_probs[:, idx] * residuals_sq)
            
            n_total += N
        
        self.sigma_sq = total_sq / n_total

    def train_on_multiple(self, datasets, iterations=20):
        """
        Iterative EM Algorithm as defined in Section 4.1., respects frozen parameters.
        """
        for _ in range(iterations):
            all_weights = []
            all_config_probs = []
            
            # E-STEP: Exact Calculation for J Stochastic Units
            for X_raw, y in datasets:
                X = self.get_X_subset(X_raw)
                N = X.shape[0]
                
                # Pre-calculate probabilities and all 2^J configurations
                probs = self.logistic(self.alpha + np.dot(X, self.a.T))
                configs = list(itertools.product([0, 1], repeat=self.J))
                
                g_vals = np.zeros((N, len(configs)))
                for idx, config in enumerate(configs):
                    # Joint probability P(I = config)
                    p_state = np.prod([probs[:, j] if config[j] == 1 else (1 - probs[:, j]) 
                                    for j in range(self.J)], axis=0)
                    
                    # Regime-specific mean
                    y_pred = self.beta_0 + np.dot(X, self.b_0)
                    for j in range(self.J):
                        if config[j] == 1:
                            y_pred += (self.beta[j] + np.dot(X, self.b[j]))
                    
                    # Normal density
                    dens = norm.pdf(y, loc=y_pred, scale=np.sqrt(self.sigma_sq))
                    g_vals[:, idx] = p_state * dens

                f_theta = np.sum(g_vals, axis=1) + 1e-9
                
                # Normalize to get P(I | y, x)
                config_probs = g_vals / f_theta[:, None]
                all_config_probs.append(config_probs)
                
                # Marginal responsibilities w_tj = E[I_tj | y_t, x_t]
                w = np.zeros((N, self.J))
                for j in range(self.J):
                    rel_idx = [idx for idx, cfg in enumerate(configs) if cfg[j] == 1]
                    w[:, j] = np.sum(config_probs[:, rel_idx], axis=1)
                all_weights.append(w)

            # M-STEP: Maximization of Expected Complete Log-Likelihood
            X_total = np.vstack([self.get_X_subset(d[0]) for d in datasets])
            W_total = np.vstack(all_weights)
            
            # Update Gating (Nonlinear) Parameters with L2 Regularization
            for j in range(self.J):
                active_a_idx = np.where(~self.frozen_mask['a'][j])[0]
                
                if len(active_a_idx) == 0:
                    continue
                
                def log_loss(active_params):
                    alpha = active_params[0]
                    full_a = self.a[j].copy()
                    full_a[active_a_idx] = active_params[1:]
                    z = alpha + np.dot(X_total, full_a)
                    p = self.logistic(z)
                    
                    # Negative log-likelihood + L2 penalty
                    nll = -np.sum(W_total[:, j] * np.log(p + 1e-9) + 
                                 (1 - W_total[:, j]) * np.log(1 - p + 1e-9))
                    penalty = 1e-4 * np.sum(active_params**2)
                    
                    return nll + penalty
                
                init_guess = np.concatenate(([self.alpha[j]], self.a[j][active_a_idx]))
                res = minimize(log_loss, init_guess, method='BFGS')
                
                self.alpha[j] = res.x[0]
                self.a[j][active_a_idx] = res.x[1:]
                self.a[j][self.frozen_mask['a'][j]] = 0

            # Update Regime (Linear) Parameters 
            self._update_linear_parameters(datasets, all_weights)
            
            # Update Noise Variance 
            self._update_sigma_squared(datasets, all_weights, all_config_probs)
    
    def freeze_parameter(self, param_type, j=None, i=None):
        """
        Freezes a parameter at zero for backward elimination.
        
        param_type: 'b_0', 'b', or 'a'
        j: hidden unit index (for 'b' and 'a')
        i: lag index
        """
        if param_type == 'b_0':
            self.b_0[i] = 0
            self.frozen_mask['b_0'][i] = True
        elif param_type == 'b':
            self.b[j, i] = 0
            self.frozen_mask['b'][j, i] = True
        elif param_type == 'a':
            self.a[j, i] = 0
            self.frozen_mask['a'][j, i] = True