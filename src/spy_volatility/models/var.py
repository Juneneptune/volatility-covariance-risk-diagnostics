from statsmodels.tsa.api import VAR
from scipy.stats import norm, t
import numpy as np
from scipy.stats import chi2


def fit_var_1(
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Input:
        - returns
    Return a dictionary:
        - "model" → fitted VARResults
        - "residuals" → residual DataFrame
        - "innovation_cov" → residual covariance (DataFrame)
    """
    filtered_returns = _filter_columns_by_suffix(returns, "_Log_Return")
    filtered_returns.columns = filtered_returns.columns.str.replace("_Log_Return", "")

    model =  VAR(filtered_returns)
    results = model.fit(1) # 1-lag
    res = results.resid
    cov = results.resid.cov()

    return {
        "Model": results,
        "residuals": res,
        "innovation_cov": cov,
    }


def gaussian_var(
    mu: float,
    sigma: float,
    alpha: float,
) -> float:
    """
    Compute the Value-at-Risk (VaR) at a specified confidence level alpha
    under the assumption that returns are normally distributed.

    Parameters:
        mu (float): Mean of the returns.
        sigma (float): Standard deviation of the returns.
        alpha (float): Significance level (e.g., 0.05 for 95% VaR).

    Returns:
        float: The VaR at the given confidence level.
    """
    return mu - sigma * norm.ppf(alpha)


def student_t_var(
    mu: float,
    sigma: float,
    nu: float,
    alpha: float,
) -> float:
    """
    Compute the Value-at-Risk (VaR) at a specified confidence level alpha
    under the assumption that returns follow a Student's t-distribution.

    Parameters:
        mu (float): Mean of the returns.
        sigma (float): Standard deviation of the returns.
        nu (float): Degrees of freedom of the Student's t-distribution.
        alpha (float): Significance level (e.g., 0.05 for 95% VaR).

    Returns:
        float: The VaR at the given confidence level.
    """
    return mu - sigma * t.ppf(alpha, nu)



def LRuc(
    alpha : float,
    x : int,
    T : int,
) -> float:
    """
    Kupiec unconditional coverage test statistic.

    Parameters:
        alpha (float): The expected violation probability (e.g., 0.01 for 99% VaR).
        x (int): Number of observed exceptions (VaR violations).
        T (int): Total number of trials (e.g., time points).

    Returns:
        float: The likelihood ratio statistic for unconditional coverage (LRuc) and p-value.
    """
    eps = 1e-12
    alpha = np.clip(alpha, eps, 1 - eps)
    p_hat = np.clip(x / T, eps, 1 - eps)

    null = (T - x) * np.log(1 - alpha) + x * np.log(alpha)
    alt  = (T - x) * np.log(1 - p_hat) + x * np.log(p_hat)

    LR = -2 * (null - alt)
    p_value = 1 - chi2.cdf(LR, df=1)
    return LR, p_value