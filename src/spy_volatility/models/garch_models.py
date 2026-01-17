from arch import arch_model
import pandas as pd
import numpy as np

class GARCHModel:
    pass

def fit_garch_11(
    returns: pd.Series,
    annualization: int = 252,
) -> pd.Series:
    """
    Fit a GARCH(1,1) model to log returns and return conditional volatility.
    Notes
    -----
    Model parameters are estimated on the full sample. The resulting conditional
    volatility series is for diagnostic comparison against realized volatility,
    not a walk-forward forecast.
    """
    garch = arch_model(
        100 * returns,  # Scaled by 100 for stable modeling (will be removed later)
        mean = "Constant",
        vol = "GARCH",
        p = 1,
        o = 0,
        q = 1,
        dist='t',
    )
    results = garch.fit(disp="off")
    cond_vol = pd.Series(np.sqrt(annualization) * results.conditional_volatility / 100, returns.index)
    cond_vol.name = "garch_11_vol"

    return cond_vol
