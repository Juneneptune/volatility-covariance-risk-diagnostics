import pandas as pd
import numpy as np

def compute_returns(
    prices: pd.DataFrame,
    price_col: list = ["SPY_Adj_Close"],
) -> pd.DataFrame:

    if isinstance(price_col, str):
        price_col = [price_col]
    # Find stock name
    for c in price_col:
        ticker = c.split("_")[0]

        # Compute log return and squared return
        log_adj_close = np.log(prices[c])
        prices[ticker + "_Log_Return"] = log_adj_close.diff()
        prices[ticker + "_Squared_Return"] = log_adj_close.diff() ** 2
    prices = prices.sort_index(axis=1)
    return prices

def compute_realized_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualization: int = 252,
) -> pd.Series:

    # Find stock name
    ticker = returns.columns[-1].split("_")[0]

    if ticker + "_Log_Return" not in returns.columns:
        raise KeyError(
            f"Required column '{ticker + "_Log_Return"}' not found. "
            f"Available columns: {list(returns.columns)}"
        )

    if ticker + "_Squared_Return" not in returns.columns:
        raise KeyError(
            f"Required column '{ticker + "_Squared_Return"}' not found. "
            f"Available columns: {list(returns.columns)}"
        )

    vol = np.sqrt(annualization * returns[ticker + "_Squared_Return"].rolling(window).mean())

    return vol
