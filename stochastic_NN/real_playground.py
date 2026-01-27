from evaluator import SNNEvaluator
from model_selector import SNNStepwiseSelector
from timeseries_processor import TimeSeriesProcessor
from real_data import RealDataExtractor
import numpy as np


# 1. Extract Data
extractor = RealDataExtractor("NVDA") # Testing on NVIDIA volatility
full_series = np.abs(extractor.get_timeseries(start_date="2020-01-01", end_date="2026-01-01"))
train_data, test_data = extractor.split_train_test(full_series)

# 2. Process (Notice we use a list for train_datasets to support your 'pooled' training)
processor = TimeSeriesProcessor(p=5)
train_datasets = [processor.create_dataset(train_data)]

# 3. Select and Train
selector = SNNStepwiseSelector(max_lags=5, max_J=2)
optimal_model = selector.run_selection(train_datasets)

# 4. Evaluate
evaluator = SNNEvaluator(optimal_model, processor, experiment_name=f"Stock Data: NVDA")
results = evaluator.evaluate(train_data, test_data)

print(f"SNN Out-of-Sample MSE: {results['SNN_Out_Sample_MSE']:.8f}")
print(f"Improvement over Linear: {results['Improvement_Pct']:.2f}%")