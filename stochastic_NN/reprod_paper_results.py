"""
Test SNN performance on the two paper examples (6.1, 6.2).

"""
import numpy as np
from time_series_generator import TimeSeriesGenerator
from timeseries_processor import TimeSeriesProcessor
from model_selector import SNNStepwiseSelector

def compute_paper_metric(model, processor, train_series, test_series, true_f_func):
    """
    This measures model error (excluding noise variance).
    Returns: average squared difference between fitted and true regression functions
    """
    # Create test dataset
    X_test_raw, y_test = processor.create_dataset(test_series)
    X_test_full = X_test_raw  
    X_test_subset = model.get_X_subset(X_test_raw)  
    
    # SNN predictions 
    f_n = model.predict_expectation(X_test_subset)
    
    # True regression function (theoretical conditional mean)
    f_true = np.array([true_f_func(X_test_full[i]) for i in range(len(X_test_full))])
    
    # Model error (excluding noise variance)
    mse = np.mean((f_n - f_true)**2)
    
    return mse

def run_example(example_name, generator_func, true_f_func, n_train, n_test, max_lags, max_J, n_runs=50):
    """
    Run one example multiple times and report statistics like Table 1.
    
    Parameters:
        example_name: String name for display
        generator_func: Function that generates one time series
        true_f_func: Function that computes true E[y_t|x_t] given x_t
        n_train: Training sample size
        n_test: Test sample size
        max_lags: Maximum lags for model selection
        max_J: Maximum hidden units
        n_runs: Number of simulation runs (paper uses 50)
    """
    
    errors = []
    selected_J_counts = {1: 0, 2: 0}
    param_counts = []
    outlier_runs = []  
    
    for run in range(n_runs):
        try:
            # Generate data
            train_series = generator_func()
            test_series = generator_func()
            
            # Fit model
            processor = TimeSeriesProcessor(p=max_lags)
            train_datasets = [processor.create_dataset(train_series)]
            selector = SNNStepwiseSelector(max_lags=max_lags, max_J=max_J)
            model = selector.run_selection(train_datasets)
            
            # Compute error using TRUE regression function
            error = compute_paper_metric(model, processor, train_series, test_series, true_f_func)
            
            # Check for outliers (after 10 runs to get baseline)
            if len(errors) >= 10:
                current_median = np.median(errors)
                if error > 10 * current_median:
                    print(f"Run {run+1}: OUTLIER (error={error:.3f}, current median={current_median:.3f})")
                    outlier_runs.append((run+1, error, model))
            
            errors.append(error)
            
            # Track model selection
            if model.J in selected_J_counts:
                selected_J_counts[model.J] += 1
            
            # Count non-zero parameters
            n_params = (1 if model.beta_0 != 0 else 0) + \
                      np.count_nonzero(model.b_0) + \
                      np.count_nonzero(model.beta) + \
                      np.count_nonzero(model.b) + \
                      np.count_nonzero(model.alpha) + \
                      np.count_nonzero(model.a) + 1  # +1 for sigma²
            param_counts.append(n_params)
            
        except Exception as e:
            print(f"Run {run+1} failed: {e}")
            continue
    
    # Compute statistics
    errors = np.array(errors)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    se_error = np.std(errors) / np.sqrt(len(errors))
    
    # Compute robust statistics (removing outliers)
    q75, q25 = np.percentile(errors, [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    non_outlier_errors = errors[errors <= outlier_threshold]
    
    mean_robust = np.mean(non_outlier_errors)
    n_outliers = len(errors) - len(non_outlier_errors)

    return {
        'mean': mean_error,
        'median': median_error,
        'mean_robust': mean_robust,
        'n_outliers': n_outliers,
        'se': se_error,
        'J_counts': selected_J_counts,
        'param_range': (np.min(param_counts), np.max(param_counts)) if param_counts else (0, 0),
        'param_avg': np.mean(param_counts) if param_counts else 0
    }

# ============================================================================
# Define true regression functions for each example
# ============================================================================

def true_f_example_61(x_t):
    """
    True regression function for Example 6.1 (Equation 18).
    """
    y_tm1 = x_t[0]  # y_{t-1}
    y_tm2 = x_t[1]  # y_{t-2}
    y_tm3 = x_t[2]  # y_{t-3}
    
    if 2*y_tm2 < y_tm1 + y_tm3:
        return 1 + 0.7*y_tm1 + 0.05*y_tm2
    else:
        return 0.8*y_tm1

def true_f_example_62(x_t):
    """
    True regression function for Example 6.2 (Equation 20).
    """
    y_tm1 = x_t[0]  # y_{t-1}
    return y_tm1 - 1 + 0.5*np.exp(1 - y_tm1)

# ============================================================================
# RUN EXAMPLES
# ============================================================================

# Example 6.1: Piecewise Linear AR
# Paper settings: n=100, k=100, x_t = (y_{t-1}, y_{t-2}, y_{t-3})
generator_61 = TimeSeriesGenerator(T=1.0, N=100)
results_61 = run_example(
    example_name="EXAMPLE 6.1: Piecewise Linear AR",
    generator_func=lambda: generator_61.generate_example_6_1(n=100),
    true_f_func=true_f_example_61,
    n_train=100,
    n_test=100,
    max_lags=3,
    max_J=2,
    n_runs=50
)

# Example 6.2: Markov Chain
# Paper settings: n=300, k=300, x_t = y_{t-1} (one-dimensional)
generator_62 = TimeSeriesGenerator(T=1.0, N=300)
results_62 = run_example(
    example_name="EXAMPLE 6.2: Markov Chain",
    generator_func=lambda: generator_62.generate_example_6_2(n=300),
    true_f_func=true_f_example_62,
    n_train=300,
    n_test=300,
    max_lags=3,
    max_J=2,
    n_runs=50
)

for key, value in results_61.items():
    print(f"Example 6.1 - {key}: {value}")

for key, value in results_62.items():
    print(f"Example 6.2 - {key}: {value}")