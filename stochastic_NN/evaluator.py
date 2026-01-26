import numpy as np
import matplotlib.pyplot as plt

class SNNEvaluator:
    def __init__(self, model, processor, experiment_name="Real Data"):
        """
        model: A trained StochasticNN instance.
        processor: The TimeSeriesProcessor used for windowing.
        experiment_name: String name of the process (e.g., 'TAR Process').
        """
        self.model = model
        self.processor = processor
        self.experiment_name = experiment_name

    def _calculate_mse(self, actual, predicted):
        return np.mean((actual - predicted)**2)

    def _fit_linear_benchmark(self, X_train, y_train):
        """
        Fits a standard Linear Autoregressive (AR) model using Least Squares.
        This provides a baseline: 'What if we assumed the world was linear?'
        """
        X_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
        # Linear Regression solution: (X'X)^-1 X'y
        coeffs, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
        return coeffs

    def _predict_linear(self, X, coeffs):
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_design, coeffs)

    def evaluate(self, train_series, test_series):
        """
        Performs in-sample and out-of-sample evaluation.
        Compares the SNN against a Linear AR model.
        """
        # 1. Prepare Data
        X_train_raw, y_train = self.processor.create_dataset(train_series)
        X_test_raw, y_test = self.processor.create_dataset(test_series)
        
        # SNN uses a subset of lags; the linear benchmark will use the same for fairness
        X_train = self.model.get_X_subset(X_train_raw)
        X_test = self.model.get_X_subset(X_test_raw)

        # 2. Train Linear Benchmark
        linear_coeffs = self._fit_linear_benchmark(X_train, y_train)

        # 3. Generate Predictions
        snn_train_pred = self.model.predict_expectation(X_train)
        snn_test_pred = self.model.predict_expectation(X_test)
        lin_test_pred = self._predict_linear(X_test, linear_coeffs)

        # 4. Calculate Metrics
        metrics = {
            "SNN_In_Sample_MSE": self._calculate_mse(y_train, snn_train_pred),
            "SNN_Out_Sample_MSE": self._calculate_mse(y_test, snn_test_pred),
            "Linear_Out_Sample_MSE": self._calculate_mse(y_test, lin_test_pred)
        }
        
        # Relative improvement: positive means SNN is better
        improvement = (metrics["Linear_Out_Sample_MSE"] - metrics["SNN_Out_Sample_MSE"]) / metrics["Linear_Out_Sample_MSE"]
        metrics["Improvement_Pct"] = improvement * 100

        # 5. Visualize
        self._plot_results(y_test, snn_test_pred, lin_test_pred)
        
        return metrics

    def _plot_results(self, actual, snn_pred, lin_pred):
        plt.figure(figsize=(12, 6))
        
        # Ground Truth
        plt.plot(actual, label=f"Actual Data ({self.experiment_name})", 
                 color='black', alpha=0.35, linewidth=2)
        
        # SNN Prediction: Represents the weighted average of the hidden regimes
        plt.plot(snn_pred, label="SNN Prediction (Non-linear Expectation)", 
                 color='crimson', linestyle='--', linewidth=1.5)
        
        # Linear AR Prediction: Represents the best-fit straight-line relationship
        plt.plot(lin_pred, label="Linear AR Benchmark (Baseline)", 
                 color='royalblue', linestyle=':', linewidth=1.5)
        
        # Caption Clarification: 
        # - Lags: The specific past time steps (y_{t-p}) chosen by Stepwise Selection.
        # - J: Number of hidden stochastic units (regime-switchers) used.
        plt.title(f"Evaluation on {self.experiment_name}\n"
                  f"Model Configuration: Lags {self.model.p_indices} | Hidden Units (J): {self.model.J}")
        
        plt.xlabel("Time Step (t)")
        plt.ylabel("Value (y)")
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def forecast_next_step(self, series):
        """
        Utility for real-time forecasting. 
        Takes the current series and predicts the very next point.
        """
        x_input_raw = self.processor.prepare_next_step_input(series)
        x_input = self.model.get_X_subset(x_input_raw)
        prediction = self.model.predict_expectation(x_input)
        return prediction[0]