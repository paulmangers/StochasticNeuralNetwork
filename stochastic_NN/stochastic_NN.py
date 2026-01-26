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
        # Standard deviation of 0.01 is enough to start the EM process
        self.beta_0 = 0.0
        self.b_0 = np.random.normal(0, 0.01, self.p)
        
        self.beta = np.random.normal(0, 0.01, J)
        self.b = np.random.normal(0, 0.01, (J, self.p))
        
        self.alpha = np.random.normal(0, 0.01, J)
        self.a = np.random.normal(0, 0.01, (J, self.p))
        
        self.sigma_sq = 1.0 # Initial variance estimate

    def logistic(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def get_X_subset(self, X_full):
        # Extracts only the selected lags from the raw windowed data
        return X_full[:, [i-1 for i in self.p_indices]]

    def predict_expectation(self, X):
        """
        Calculates E[y_t | x_t] as defined in Section 2.2
        """
        y_hat = self.beta_0 + np.dot(X, self.b_0)
        probs = self.logistic(self.alpha + np.dot(X, self.a.T))
        for j in range(self.J):
            y_hat += (self.beta[j] + np.dot(X, self.b[j])) * probs[:, j]
        return y_hat

    def _update_linear_parameters(self, datasets, all_weights):
        """
        Full implementation of Section 4.1 linear parameter updates.
        Solves the (J+1)*(p+1) system of equations simultaneously while
        respecting pruned (zero) parameters.
        """
        X_list = [self.get_X_subset(d[0]) for d in datasets]
        y_total = np.concatenate([d[1] for d in datasets])
        X_total = np.vstack(X_list)
        W_total = np.vstack(all_weights)
        
        N_total = len(y_total)
        dim = self.p + 1 
        total_dim = (self.J + 1) * dim
        
        M = np.zeros((total_dim, total_dim))
        V = np.zeros(total_dim)
        X_star = np.column_stack([np.ones(N_total), X_total])
        
        # --- Build V (Right-hand side) ---
        V[:dim] = np.dot(X_star.T, y_total)
        for j in range(self.J):
            V[(j+1)*dim : (j+2)*dim] = np.dot((X_star * W_total[:, j][:, None]).T, y_total)
            
        # --- Build M (Left-hand side Block Matrix) ---
        M[:dim, :dim] = np.dot(X_star.T, X_star)
        for j in range(self.J):
            W_j = W_total[:, j][:, None]
            weighted_XTX = np.dot((X_star * W_j).T, X_star)
            M[:dim, (j+1)*dim : (j+2)*dim] = weighted_XTX
            M[(j+1)*dim : (j+2)*dim, :dim] = weighted_XTX
            M[(j+1)*dim : (j+2)*dim, (j+1)*dim : (j+2)*dim] = weighted_XTX

        # --- Constrained Solving (Respecting Pruned Parameters) ---
        # Flatten current parameters to find non-zero indices
        current_theta = np.concatenate(([self.beta_0], self.b_0))
        for j in range(self.J):
            current_theta = np.concatenate((current_theta, [self.beta[j]], self.b[j]))
        
        active_mask = np.where(current_theta != 0)[0]
        
        # Reduced system
        M_active = M[np.ix_(active_mask, active_mask)]
        V_active = V[active_mask]
        
        try:
            Theta_reduced = np.linalg.solve(M_active, V_active)
        except np.linalg.LinAlgError:
            Theta_reduced = np.linalg.lstsq(M_active, V_active, rcond=None)[0]
            
        Theta_full = np.zeros(total_dim)
        Theta_full[active_mask] = Theta_reduced

        # Distribute results
        self.beta_0 = Theta_full[0]
        self.b_0 = Theta_full[1:dim]
        for j in range(self.J):
            start = (j + 1) * dim
            self.beta[j] = Theta_full[start]
            self.b[j] = Theta_full[start+1 : start+dim]

    def train_on_multiple(self, datasets, iterations=20):
        """
        Iterative EM Algorithm as defined in Section 4.1.
        Includes pooled series processing and multi-state E-step.
        """
        for _ in range(iterations):
            all_weights = []
            
            # --- E-STEP: Exact Calculation for J Stochastic Units ---
            for X_raw, y in datasets:
                X = self.get_X_subset(X_raw)
                N = X.shape[0]
                
                # Pre-calculate probabilities and all 2^J configurations
                probs = self.logistic(self.alpha + np.dot(X, self.a.T))
                configs = list(itertools.product([0, 1], repeat=self.J))
                
                g_vals = np.zeros((N, len(configs)))
                for idx, config in enumerate(configs):
                    # Joint probability P(I)
                    p_state = np.prod([probs[:, j] if config[j] == 1 else (1 - probs[:, j]) 
                                    for j in range(self.J)], axis=0)
                    
                    # Regime-specific mean for config
                    y_pred = self.beta_0 + np.dot(X, self.b_0)
                    for j in range(self.J):
                        if config[j] == 1:
                            y_pred += (self.beta[j] + np.dot(X, self.b[j]))
                    
                    # Normal density: sigma^-1 * phi(...)
                    dens = norm.pdf(y, loc=y_pred, scale=np.sqrt(self.sigma_sq))
                    g_vals[:, idx] = p_state * dens

                f_theta = np.sum(g_vals, axis=1) + 1e-9
                
                # Responsibilities w_tj
                w = np.zeros((N, self.J))
                for j in range(self.J):
                    rel_idx = [idx for idx, cfg in enumerate(configs) if cfg[j] == 1]
                    w[:, j] = np.sum(g_vals[:, rel_idx], axis=1) / f_theta
                all_weights.append(w)

            # --- M-STEP: Maximization of Expected Complete Log-Likelihood ---
            X_total = np.vstack([self.get_X_subset(d[0]) for d in datasets])
            W_total = np.vstack(all_weights)
            
            # 1. Update Gating (Nonlinear) Parameters with Pruning Constraints
            for j in range(self.J):
                active_a_idx = np.where(self.a[j] != 0)[0]
                
                def log_loss(active_params):
                    alpha = active_params[0]
                    full_a = np.zeros(self.p)
                    full_a[active_a_idx] = active_params[1:]
                    z = alpha + np.dot(X_total, full_a)
                    p = self.logistic(z)
                    return -np.sum(W_total[:, j] * np.log(p + 1e-9) + 
                                  (1 - W_total[:, j]) * np.log(1 - p + 1e-9))
                
                init_guess = np.concatenate(([self.alpha[j]], self.a[j][active_a_idx]))
                res = minimize(log_loss, init_guess, method='BFGS')
                self.alpha[j] = res.x[0]
                self.a[j][active_a_idx] = res.x[1:]

            # 2. Update Regime (Linear) Parameters
            self._update_linear_parameters(datasets, all_weights)
            
            # 3. Update Noise Variance sigma^2
            resids_sq = []
            for X_raw, y in datasets:
                y_hat = self.predict_expectation(self.get_X_subset(X_raw))
                resids_sq.append((y - y_hat)**2)
            self.sigma_sq = np.mean(np.concatenate(resids_sq))