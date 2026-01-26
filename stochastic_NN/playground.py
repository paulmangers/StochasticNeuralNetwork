import numpy as np
import matplotlib.pyplot as plt
from stochastic_NN import StochasticNN
from model_selector import SNNStepwiseSelector
from time_series_generator import TimeSeriesGenerator
from timeseries_processor import TimeSeriesProcessor
from evaluator import SNNEvaluator

# --- DEBUGGING WRAPPER ---
def run_experiment(M=3, T=1.0, N=300, max_lags=3):
    print("Initializing Generators and Processors...")
    generator = TimeSeriesGenerator(T, N)
    processor = TimeSeriesProcessor(max_lags)
    
    experiments = [
        {
            "name": "TAR Process",
            "func": lambda: generator.generate_tar(z0=0.1, threshold=0.0)
        },
        {
            "name": "Logistic Map",
            "func": lambda: generator.generate_logistic(z0=0.4, r=3.9)
        },
        {
            "name": "Ito Process",
            "func": lambda: generator.ito_process(z0=0.4, mufunc=lambda t, z: 0.5 * z, sigfunc=lambda t, z: 0.2 * z)
        }

    ]

    for exp in experiments:
        print(f"\n>>> Starting Experiment: {exp['name']}")
        
        # 1. Generate Data
        print(f"Generating {M} paths...")
        train_paths = [exp['func']() for _ in range(M)]
        train_datasets = [processor.create_dataset(p) for p in train_paths]
        print("Data generation successful.")

        # 2. Model Selection
        print("Entering Stepwise Selector...")
        selector = SNNStepwiseSelector(max_lags=max_lags, max_J=1) # Start small for debugging
        
        try:
            optimal_model = selector.run_selection(train_datasets)
            print(f"Selection Complete. Optimal Lags: {optimal_model.p_indices}")
        except Exception as e:
            print(f"CRITICAL ERROR during selection: {e}")
            continue

        # 3. Evaluation
        print("Evaluating Performance...")
        evaluator = SNNEvaluator(optimal_model, processor, experiment_name=exp['name'])
        
        test_path = exp['func']()
        results = evaluator.evaluate(train_paths[0], test_path)
        print(f"\n--- Statistical Comparison for {exp['name']} ---")
        print(f"Total Out-of-Sample Points: {len(test_path) - max_lags}")
        print(f"Linear Benchmark Loss:      {results['Linear_Out_Sample_MSE']:.6f}")
        print(f"SNN (Proposed) Loss:       {results['SNN_Out_Sample_MSE']:.6f}")
        print(f"Reduction in Variance:     {results['Improvement_Pct']:.2f}%")
        if results['Improvement_Pct'] > 0:
            print("Conclusion: The Stochastic neurons successfully captured non-linearities.")
        else:
            print("Conclusion: Linear model is sufficient; SNN is overfitting.")
        print(f"Experiment {exp['name']} Finished.")
        print(f"Result: {results['Improvement_Pct']:.2f}% improvement.")

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"The program crashed with error: {e}")

