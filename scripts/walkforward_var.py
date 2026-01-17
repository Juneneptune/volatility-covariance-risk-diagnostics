from spy_volatility.data.loaders import load_or_update_spy_prices
from spy_volatility.models.garch_models import fit_garch_11
from spy_volatility.utils.config import load_config, get_project_root
from spy_volatility.data.features import compute_returns, compute_realized_volatility
from spy_volatility.models.var import gaussian_var, student_t_var, LRuc
from spy_volatility.risk.cov_metrics import rolling_sample_covariance, covariance_diagnostics
from spy_volatility.risk.spd import try_cholesky, add_jitter, clip_eigenvalues
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    # Get configs, SPY data, and compute returns
    root = get_project_root()
    cfg = load_config("default.yaml")
    prices = load_or_update_spy_prices(cfg, allow_data_update=False)
    returns = compute_returns(prices, price_col=prices.columns).dropna()

    # Compute realized volatility and garch predicted volatility
    garch = fit_garch_11(returns["SPY_Log_Return"], annualization=1).dropna()
    rv252 = compute_realized_volatility(returns, window=252, annualization=1).dropna()

    ### Compute VaR 0.01 and 0.05 for both normal and student-t distribution ###

    var_results = returns[["SPY_Log_Return"]].copy()

    # RV — Gaussian
    var_results["exceed_rv_gauss_001"] = 0
    var_results["exceed_rv_gauss_005"] = 0

    # RV — Student-t
    var_results["exceed_rv_t_001"] = 0
    var_results["exceed_rv_t_005"] = 0

    # GARCH — Gaussian
    var_results["exceed_garch_gauss_001"] = 0
    var_results["exceed_garch_gauss_005"] = 0

    # GARCH — Student-t
    var_results["exceed_garch_t_001"] = 0
    var_results["exceed_garch_t_005"] = 0


    # Walk-forward exceedance computation
    for date in rv252.index[:-1]:
        date_idx = returns.index.get_loc(date) + 1
        future_date = returns.index[date_idx]
        future_return = returns.iloc[date_idx]["SPY_Log_Return"]

        rv_sigma = rv252[date]
        garch_sigma = garch[date]

        var_results.loc[future_date, "exceed_rv_gauss_001"] = int(
            future_return < gaussian_var(mu=0, sigma=rv_sigma, alpha=0.99)
        )
        var_results.loc[future_date, "exceed_rv_gauss_005"] = int(
            future_return < gaussian_var(mu=0, sigma=rv_sigma, alpha=0.95)
        )


        var_results.loc[future_date, "exceed_rv_t_001"] = int(
            future_return < student_t_var(mu=0, nu=8, sigma=rv_sigma, alpha=0.99)
        )
        var_results.loc[future_date, "exceed_rv_t_005"] = int(
            future_return < student_t_var(mu=0, nu=8, sigma=rv_sigma, alpha=0.95)
        )


        var_results.loc[future_date, "exceed_garch_gauss_001"] = int(
            future_return < gaussian_var(mu=0, sigma=garch_sigma, alpha=0.99)
        )
        var_results.loc[future_date, "exceed_garch_gauss_005"] = int(
            future_return < gaussian_var(mu=0, sigma=garch_sigma, alpha=0.95)
        )

        var_results.loc[future_date, "exceed_garch_t_001"] = int(
            future_return < student_t_var(mu=0, nu=8, sigma=garch_sigma, alpha=0.99)
        )
        var_results.loc[future_date, "exceed_garch_t_005"] = int(
            future_return < student_t_var(mu=0, nu=8, sigma=garch_sigma, alpha=0.95)
        )

    # print(var_results.tail)

    # Empirical exceedance rates
    T = len(var_results)
    results = []
    for col in [
        "exceed_rv_gauss_001",
        "exceed_rv_gauss_005",
        "exceed_rv_t_001",
        "exceed_rv_t_005",
        "exceed_garch_gauss_001",
        "exceed_garch_gauss_005",
        "exceed_garch_t_001",
        "exceed_garch_t_005",
    ]:
        exceed_rate = var_results[col].mean()
        x = var_results[col].sum()
        
        alpha = 0.01 if col.endswith("001") else 0.05

        # Kupiec unconditional coverage test statistic
        lruc, p_value = LRuc(alpha=alpha, x=int(x), T=T)

        results.append(
            {
                "model": col.split("_")[1],          # rv / garch
                "distribution": col.split("_")[2],   # gauss / t
                "alpha": alpha,
                "exceedance_rate": exceed_rate,
                "LRuc p-value": p_value,
            }
        )

    results = pd.DataFrame(results)


    # Plot for alpha 0.99 on GARCH
    alpha = 0.99

    # Align series
    plot_idx = garch.index.intersection(returns.index)

    ret = returns.loc[plot_idx, "SPY_Log_Return"]
    sigma = garch.loc[plot_idx]

    # VaR series
    var_gauss = pd.Series(
        gaussian_var(mu=0, sigma=sigma, alpha=alpha),
        index=plot_idx,
    )

    var_t = pd.Series(
        student_t_var(mu=0, nu=8, sigma=sigma, alpha=alpha),
        index=plot_idx,
    )

    exceed_gauss = var_results.loc[plot_idx, "exceed_garch_gauss_001"]
    exceed_t = var_results.loc[plot_idx, "exceed_garch_t_001"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(ret.index, ret, color="black", lw=0.8, label="Returns")
    ax.plot(var_gauss.index, var_gauss, color="blue", lw=1.5, label="VaR (Gaussian 99%)")
    ax.plot(var_t.index, var_t, color="orange", lw=1.5, label="VaR (Student-t 99%)")

    ax.scatter(
        ret.index[exceed_gauss == 1],
        ret[exceed_gauss == 1],
        color="blue",
        marker="x",
        s=30,
        label="Gaussian exceedance",
        zorder=3,
    )

    ax.scatter(
        ret.index[exceed_t == 1],
        ret[exceed_t == 1],
        facecolors="none",
        edgecolors="red",
        marker="o",
        s=40,
        label="Student-t exceedance",
        zorder=3,
    )

    ax.set_title("SPY 1-Day VaR Backtest (GARCH, 99%)")
    ax.set_ylabel("Log Return")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{root}/data/outputs/figures/var_garch_99.png", dpi=200)
    plt.close()

    # plot LRuc table
    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(results)))
    ax.axis("off")

    cellText = results.applymap(
        lambda x: f"{x:.4g}" if isinstance(x, (int, float)) else x
    ).values # Get 4 most significant digits

    table = ax.table(
        cellText=cellText,
        colLabels=results.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    ax.set_title("Kupiec Unconditional Coverage Test (SPY)", pad=20)

    plt.tight_layout()
    plt.savefig(f"{root}/data/outputs/figures/var_lruc_table.png", dpi=200)
    plt.close()

    # Clustering check -> rolling mean on exceedance

    rolling_mean_exceed = (
        var_results["exceed_garch_gauss_005"]
        .rolling(63)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        rolling_mean_exceed.index,
        rolling_mean_exceed,
        label="Rolling exceedance rate (GARCH, 5%)",
        linewidth=2,
    )

    ax.axhline(
        0.05,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Target α = 5%",
    )

    ax.set_title("Exceedance Clustering (63-Day Rolling Window)")
    ax.set_ylabel("Exceedance Rate")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{root}/data/outputs/figures/exceedance_clustering.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()