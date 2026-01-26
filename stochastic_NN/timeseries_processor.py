import numpy as np

class TimeSeriesProcessor:
    def __init__(self, p):
        """
        p: Number of lags to use for the input vector x_t.
        """
        self.p = p

    def create_dataset(self, series):
        """
        Transforms a vector [z_0, ..., z_T] into X and y.
        X[t] = [z_{t-1}, ..., z_{t-p}]
        y[t] = z_t
        """
        X, y = [], []
        for t in range(self.p, len(series)):
            X.append(series[t-self.p:t][::-1]) # [y_{t-1}, ..., y_{t-p}]
            y.append(series[t])
        return np.array(X), np.array(y)

    def prepare_next_step_input(self, series):
        """
        Takes the most recent values to predict the future z_{t+1}.
        """
        return np.array(series[-self.p:][::-1]).reshape(1, -1)