from real_data import RealDataExtractor
from timeseries_processor import TimeSeriesProcessor
from model_selector import SNNStepwiseSelector
from evaluator import SNNEvaluator
import numpy as np

# Extract returns
extractor = RealDataExtractor('^GSPC')
returns = extractor.get_timeseries(start_date="2018-01-01", end_date="2025-01-28")

# Compute log(absolute returns)
volatility = np.log(np.abs(returns)*100 + 1e-8) 

# Split data
train_data, test_data = extractor.split_train_test(volatility, train_ratio=0.8)

# Fit SNN
processor = TimeSeriesProcessor(p=5)
train_datasets = [processor.create_dataset(train_data)]

selector = SNNStepwiseSelector(max_lags=5, max_J=2)
optimal_model = selector.run_selection(train_datasets)

# Evaluate
evaluator = SNNEvaluator(optimal_model, processor, experiment_name="S&P 500 Volatility")
results = evaluator.evaluate(train_data, test_data)

# Results

print(f"  Selected: J={optimal_model.J}, Lags={optimal_model.p_indices}")
print(f"  Out-of-Sample MSE: {results['SNN_Out_Sample_MSE']:.6f}")
print(f"  Linear Benchmark MSE: {results['Linear_Out_Sample_MSE']:.6f}")