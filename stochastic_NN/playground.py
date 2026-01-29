import numpy as np
import matplotlib.pyplot as plt
from stochastic_NN import StochasticNN
from model_selector import SNNStepwiseSelector
from time_series_generator import TimeSeriesGenerator
from timeseries_processor import TimeSeriesProcessor
from evaluator import SNNEvaluator


def run_experiment(M=10, T=1.0, N=500, max_lags=3):
    '''
    Runs a full experiment comparing SNN to Linear AR on various time series.
    
    M: Number of training paths 
    T: Time horizon 
    N: Number of time steps
    max_lags: Maximum lags for model selection
    '''
    generator = TimeSeriesGenerator(T, N)
    processor = TimeSeriesProcessor(max_lags)

    experiments = [
        {
            "name": "TAR Process",
            "func": lambda: generator.generate_tar(z0=0.1, threshold=0.0)
        },      
        {
            "name": "3 Regime Market Model",
            "func": lambda: generator.generate_bull_bear_sideways()
        },
        {
            "name": "Logistic Map",
            "func": lambda: generator.generate_logistic(z0=0.4, r=3.9)
        },
        {
            "name": "Ito Process",
            "func": lambda: generator.ito_process(z0=0.4, mufunc=lambda t, z: 0.05 * z, sigfunc=lambda t, z: 0.2 * z)
        }

    ]

    for exp in experiments:
        # Generate training data
        train_paths = [exp['func']() for _ in range(M)]
        train_datasets = [processor.create_dataset(p) for p in train_paths]
        # Model Selection
        selector = SNNStepwiseSelector(max_lags=max_lags, max_J=3)
        try:
            optimal_model = selector.run_selection(train_datasets)
        except Exception as e:
            print(f"CRITICAL ERROR during selection: {e}")
            continue

        # Evaluation
        evaluator = SNNEvaluator(optimal_model, processor, experiment_name=exp['name'])
        # Generate test data
        test_path = exp['func']()

        results = evaluator.evaluate(train_paths[0], test_path)
        print(f"\n--- Statistical Comparison for {exp['name']} ---")
        print(f"Total Out-of-Sample Points: {len(test_path) - max_lags}")
        print(f"Benchmark Loss:      {results['Linear_Out_Sample_MSE']:.6f}")
        print(f"SNN Loss:       {results['SNN_Out_Sample_MSE']:.6f}")

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"The program crashed with error: {e}")

