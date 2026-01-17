from spy_volatility.data.loaders import load_or_update_spy_prices, load_or_update_prices
from spy_volatility.models.garch_models import fit_garch_11
from spy_volatility.utils.config import load_config, get_project_root
from spy_volatility.data.features import compute_returns, compute_realized_volatility
from spy_volatility.models.var import gaussian_var, student_t_var, LRuc
from spy_volatility.risk.cov_metrics import rolling_sample_covariance, covariance_diagnostics
from spy_volatility.risk.spd import try_cholesky, add_jitter, clip_eigenvalues
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

def main() -> None:

    root = get_project_root()

    # Load SPY and cfg
    cfg = load_config("default.yaml")
    prices = load_or_update_spy_prices(cfg, allow_data_update=False)

    # Load multivariate assets
    mult_cfg = load_config("default_multivar.yaml")
    mult_prices = load_or_update_prices(mult_cfg, allow_data_update=False, show_only_adj_close=True)

    # Compute returns, rolling volatility, and realized covariance
    returns = compute_returns(prices, price_col=["SPY_Adj_Close"]).dropna()
    mult_returns = compute_returns(mult_prices, price_col=mult_prices.columns).dropna()

    spy_rv21 = compute_realized_volatility(returns, window=21, annualization=1).dropna()
    mult_rc21 = rolling_sample_covariance(mult_returns, window=21)

    ### Label regime by high volatility for 70th percentile RV
    rv_threshold = spy_rv21.quantile(0.70)
    # Use RV_t to define regime applied to return at t+1
    regime = []
    for date in spy_rv21.index.intersection(mult_rc21.keys()):
        regime.append({
            "date": date,
            "high_vol": int(spy_rv21.loc[date] > rv_threshold),
            "rv21": spy_rv21.loc[date],
        })

    
    regime_df = pd.DataFrame(regime).set_index("date")

    # Diagnose covariance
    diag = []
    for date in regime_df.index:
        cov_diag = covariance_diagnostics(mult_rc21[date])
        diag.append({
            "date": date,
            "min_eigenvalue": cov_diag["min_eigenvalue"],
            "max_eigenvalue": cov_diag["max_eigenvalue"],
            "condition_number": cov_diag["condition_number"],
        })
    diag_df = pd.DataFrame(diag).set_index("date")
    
    # Combine regime + diagnostics
    regime_diag_df = pd.concat([diag_df, regime_df], axis=1)

    # Plot
    df = regime_diag_df.copy()

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # Top panel: RV(21) + threshold
    axes[0].plot(
        df.index,
        df["rv21"],
        color="black",
        linewidth=1.2,
    )

    axes[0].axhline(
        rv_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
    )

    axes[0].set_ylabel("RV(21)")
    axes[0].set_title("SPY Realized Volatility with High-Vol Threshold (70th Percentile)")


    # Bottom panel: Condition number
    axes[1].plot(
        df.index,
        df["condition_number"],
        color="tab:blue",
        linewidth=1.2,
    )

    axes[1].set_ylabel("Condition Number")
    axes[1].set_title("Covariance Conditioning (Rolling RC21)")


    # Shade high-vol regimes
    in_regime = df["high_vol"].values

    start = None
    for i in range(len(df)):
        if in_regime[i] and start is None:
            start = df.index[i]
        elif not in_regime[i] and start is not None:
            axes[0].axvspan(start, df.index[i], color="red", alpha=0.15)
            axes[1].axvspan(start, df.index[i], color="red", alpha=0.15)
            start = None

    # Handle case where regime continues to the end
    if start is not None:
        axes[0].axvspan(start, df.index[-1], color="red", alpha=0.15)
        axes[1].axvspan(start, df.index[-1], color="red", alpha=0.15)

    plt.tight_layout()
    plt.savefig(f"{root}/data/outputs/figures/vol_regimes_diagnostic.png", dpi=150)
    plt.close()
    

if __name__ == "__main__":
    main()