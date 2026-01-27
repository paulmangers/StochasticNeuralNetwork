import yfinance as yf
import numpy as np

class RealDataExtractor:
    def __init__(self, ticker):
        """
        ticker: String symbol (e.g., 'NVDA', 'BTC-USD', '^GSPC' for S&P 500)
        """
        self.ticker = ticker

    def get_timeseries(self, start_date="2020-01-01", end_date="2026-01-01", use_returns=True):
        print(f"Fetching data for {self.ticker}...")
        
        # We add auto_adjust=True to merge 'Close' and 'Adj Close'
        data = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"No data found for {self.ticker}.")

        # Robust selection: take the 'Close' column
        # Depending on yfinance version, this handles the MultiIndex issue
        if 'Close' in data.columns:
            prices = data['Close'].values.flatten()
        else:
            # Fallback: take the first column available
            prices = data.iloc[:, 0].values.flatten()

        # Remove any potential NaNs (common in real-world data)
        prices = prices[~np.isnan(prices)]

        if use_returns:
            # Log-returns: log(P_t) - log(P_{t-1})
            returns = np.diff(np.log(prices))
            return returns
        
        return prices

    def split_train_test(self, data, train_ratio=0.8):
        """
        Splits the series into training and testing sets.
        """
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]