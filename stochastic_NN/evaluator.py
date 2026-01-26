import numpy as np
import matplotlib.pyplot as plt

class SNNEvaluator:
    def __init__(self, model, processor):
        """
        model: A trained StochasticNN instance
        processor: The TimeSeriesProcessor used for windowing
        """
        self.model = model
        self.processor = processor

    def _calculate_mse(self, actual, predicted):
        return np.mean((actual - predicted)**2)

    def _calculate_mae(self, actual, predicted):
        return np.mean(np.abs(actual - predicted))

    def _fit_linear_benchmark(self, X_train, y_train):
        """
        Fits a standard Linear Autoregressive (AR) model using Least Squares.
        This serves as the 'Baseline' to prove the SNN's value.
        """
        # Add intercept column
        X_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
        # Solve (X^T * X) * theta = X^T * y
        coeffs = np.linalg.lstsq(X_design, y_train, rcond=None)[0]
        return coeffs

    def _predict_linear(self, X, coeffs):
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_design, coeffs)

    def evaluate(self, train_series, test_series):
        """
        Evaluates PMSE on training (in-sample) and test (out-of-sample) data.
        """
        # Prepare Data
        X_train_raw, y_train = self.processor.create_dataset(train_series)
        X_test_raw, y_test = self.processor.create_dataset(test_series)
        
        # Ensure we only use the lags selected by the SNN
        X_train = self.model.get_X_subset(X_train_raw)
        X_test = self.model.get_X_subset(X_test_raw)

        # 1. Train Linear Benchmark
        linear_coeffs = self._fit_linear_benchmark(X_train, y_train)

        # 2. Generate Predictions
        snn_train_pred = self.model.predict_expectation(X_train)
        snn_test_pred = self.model.predict_expectation(X_test)
        
        lin_test_pred = self._predict_linear(X_test, linear_coeffs)

        # 3. Calculate Metrics (Predictive Mean Squared Error)
        metrics = {
            "SNN_In_Sample_MSE": self._calculate_mse(y_train, snn_train_pred),
            "SNN_Out_Sample_MSE": self._calculate_mse(y_test, snn_test_pred),
            "Linear_Out_Sample_MSE": self._calculate_mse(y_test, lin_test_pred)
        }

        # Calculate "Improvement over Linear" as a percentage
        improvement = (metrics["Linear_Out_Sample_MSE"] - metrics["SNN_Out_Sample_MSE"]) / metrics["Linear_Out_Sample_MSE"]
        metrics["Improvement_Pct"] = improvement * 100

        self._plot_results(y_test, snn_test_pred, lin_test_pred)
        
        return metrics

    def _plot_results(self, actual, snn_pred, lin_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Actual Data", color='black', alpha=0.4, linewidth=2)
        plt.plot(snn_pred, label="SNN Prediction (Stochastic/Non-linear)", color='red', linestyle='--')
        plt.plot(lin_pred, label="AR Prediction (Linear Benchmark)", color='blue', linestyle=':')
        plt.title(f"Performance: SNN vs Linear AR (Lags: {self.model.p_indices}, J: {self.model.J})")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def forecast_next_step(self, series):
        """
        The final prediction tool: Input raw history, output z_{t+1}.
        """
        x_input_raw = self.processor.prepare_next_step_input(series)
        x_input = self.model.get_X_subset(x_input_raw)
        prediction = self.model.predict_expectation(x_input)
        return prediction[0]