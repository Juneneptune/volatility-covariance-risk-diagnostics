# src/spy_volatility/data/loaders.py

import datetime as dt
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yfinance as yf

from spy_volatility.utils.config import get_project_root

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten yfinance multi-index columns.
    """
    df.columns = ['_'.join(c.replace(" ", "_") for c in col) 
        if isinstance(col, tuple) else col.replace(" ", "_") 
        for col in df.columns] # Flatten multi-index: join levels with underscore
    return df

def _resolve_spy_prices_path(cfg: Dict[str, Any]) -> Path:
    """
    Use the config to compute the full path to the SPY prices CSV.

    If cfg["data"]["spy_prices_file"] == "data/spy/spy_prices.csv"
    and project_root is spy-volatility-clustering-and-garch,
    the full path is spy-volatility-clustering-and-garch/data/spy/spy_prices.csv
    """
    project_root = get_project_root()
    rel_path = cfg["data"]["spy_prices_file"]
    full_path = project_root / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path

def _download_spy_prices(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Download SPY data from Yahoo Finance based on start_date and end_date in config.

    Config entries used:
      cfg["data"]["spy_ticker"]
      cfg["data"]["start_date"]
      cfg["data"]["end_date"]  (if null, we use today's date)
    Returns a DataFrame indexed by date with flattened column names.
    """
    ticker = cfg["data"]["spy_ticker"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]

    if end_date is None:
        end_date = dt.date.today().isoformat()

    print(f"[loaders] Downloading {ticker} from {start_date} to {end_date} ...")

    spy = yf.download(ticker, start=start_date, end=end_date, group_by=None, auto_adjust=False)

    if df.empty:
        raise RuntimeError("[loaders] No data returned from yfinance.")
    df = _flatten_columns(df)
    df = df.sort_index()
    return df

def load_or_update_spy_prices(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Main entry point for SPY prices.

    Behavior:
      - Compute CSV path from config
      - If file does not exist:
          * download full history (start_date -> end_date)
          * save to CSV
          * return DataFrame
      - If file exists:
          * load CSV
          * find last available date
          * download new data from last_date+1 to end_date (from config, or today)
          * append new rows, remove duplicates
          * save back to CSV
          * return updated DataFrame
    """
    spy_csv_path = _resolve_spy_prices_path(cfg)

    if not spy_csv_path.exists():
        # No file yet: full download
        print(f"[load_or_update_spy] {spy_csv_path} not found. Downloading full history.")
        spy = _download_spy_prices(cfg)
        spy.to_csv(spy_csv_path)
        print(f"[load_or_update_spy] Saved to {spy_csv_path}")
        return spy

    # File exists: load and update
    print(f"[load_or_update_spy] Loading existing SPY data from {spy_csv_path}")
    spy_old = pd.read_csv(spy_csv_path, parse_dates=[0], index_col=0)
    spy_old = spy_old.sort_index()

    last_date = spy_old.index.max()
    print(f"[load_or_update_spy] Last date in file: {last_date.date()}")

    # Determine end date
    end_date = cfg["data"]["end_date"]
    if end_date is None:
        end_date = dt.date.today().isoformat()

    # Compute next start date = last_date + 1 day
    next_start = (last_date + pd.Timedelta(days=1)).date().isoformat()
    print(f"[load_or_update_spy] Downloading updates from {next_start} to {end_date} ...")

    # Temporary config for update range
    cfg_update = cfg.copy()
    cfg_update["data"] = cfg["data"].copy()
    cfg_update["data"]["start_date"] = next_start
    cfg_update["data"]["end_date"] = end_date

    spy_new = _download_spy_prices(cfg_update)

    if spy_new.empty:
        print("[load_or_update_spy] No new rows. Returning existing data.")
        return spy_old

    # Append and drop duplicates
    spy = pd.concat([spy_old, spy_new])
    spy = spy[~spy.index.duplicated(keep="last")]
    spy = spy.sort_index()

    spy.to_csv(spy_csv_path)
    print(f"[load_or_update_spy] Updated data saved to {spy_csv_path}")
    return spy