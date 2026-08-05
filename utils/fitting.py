import numpy as np
import streamlit as st
from scipy.optimize import curve_fit, approx_fprime
from scipy.stats import t
import pandas as pd

def compute_confidence_intervals(time, params, covariance, alpha, dof, residual_variance, model_func):
    """
    Lines ~1034-1100: Compute confidence intervals for fitted data with approx_fprime.
    """
    t_critical=t.ppf(1-alpha/2,dof)
    epsilon=np.sqrt(np.finfo(float).eps)
    conf_interval=np.zeros(len(time))
    fitted_values=model_func(time,*params)

    for i in range(len(time)):
        def func(p):
            return model_func(np.array([time[i]]),*p)
        gradient=approx_fprime(params, func, epsilon)
        conf_interval[i]=np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T)))

    lower_bound=fitted_values - t_critical*conf_interval
    upper_bound=fitted_values + t_critical*conf_interval
    return lower_bound, upper_bound


def compute_fit_statistics(time_vals, y_data, popt, pcov, model_func, param_names, alpha=0.05):
    """
    Compute the post-fit statistics shared by the manual and automatic fitting paths.

    Returns a dict of results, or None when there are not enough data points to
    support the number of fitted parameters (dof <= 0).
    """
    n_points = len(y_data)
    n_params = len(popt)
    dof = n_points - n_params
    if dof <= 0:
        return None

    y_pred = model_func(time_vals, *popt)
    residuals = y_data - y_pred
    residual_variance = np.var(residuals, ddof=n_params)

    RSS = np.sum(residuals ** 2)
    TSS = np.sum((y_data - np.mean(y_data)) ** 2)
    R_squared = 1 - (RSS / TSS) if TSS > 0 else np.nan
    AIC = 2 * n_params + n_points * np.log(RSS / n_points) if RSS > 0 else -np.inf

    perr = np.sqrt(np.diag(pcov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = popt / perr

    lower_bound, upper_bound = compute_confidence_intervals(
        time_vals, popt, pcov, alpha, dof, residual_variance, model_func
    )

    data_variance = np.var(y_data)
    variance_ratio = residual_variance / data_variance if data_variance > 0 else np.inf
    poor_fit = bool(variance_ratio > 0.2 or R_squared < 0.90)

    raw_p_values = 2 * (1 - t.cdf(np.abs(t_statistic), df=dof))
    # Poor fits get a deliberately high p-value so they are not read as significant.
    p_values = [0.5 for _ in popt] if poor_fit else raw_p_values

    results = {
        "y_pred": y_pred,
        "residuals": residuals,
        "residual_variance": residual_variance,
        "dof": dof,
        "RSS": RSS,
        "TSS": TSS,
        "R_squared": R_squared,
        "AIC": AIC,
        "param_errors": perr,
        "t_statistic": t_statistic,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "variance_ratio": variance_ratio,
        "poor_fit": poor_fit,
        "raw_p_values": raw_p_values,
        "p_values": p_values,
        "lag_time": None,
        "lag_time_std_err": None,
    }

    if "q0" in param_names and "mu" in param_names:
        q0_idx = param_names.index("q0")
        mu_idx = param_names.index("mu")
        lag_time, lag_time_std_err = compute_lag_time(
            q0=popt[q0_idx],
            mu=popt[mu_idx],
            q0_std_err=perr[q0_idx],
            mu_std_err=perr[mu_idx],
        )
        results["lag_time"] = lag_time
        results["lag_time_std_err"] = lag_time_std_err

    return results


def compute_lag_time(q0, mu, q0_std_err=None, mu_std_err=None):
    """
    Compute lag time and optional uncertainty from Baranyi-style parameters.

    T_lag = ln(1 + 1/q0) / mu
    """
    if q0 is None or mu is None:
        return None, None

    if not np.isfinite(q0) or not np.isfinite(mu) or q0 <= 0 or mu <= 0:
        return None, None

    lag_time = np.log(1.0 + (1.0 / q0)) / mu
    lag_time_std_err = None

    if q0_std_err is not None and mu_std_err is not None:
        if np.isfinite(q0_std_err) and np.isfinite(mu_std_err) and q0_std_err >= 0 and mu_std_err >= 0:
            # First-order propagation without covariance term.
            dlag_dq0 = -1.0 / (mu * q0 * (q0 + 1.0))
            dlag_dmu = -np.log(1.0 + (1.0 / q0)) / (mu ** 2)
            lag_var = (dlag_dq0 ** 2) * (q0_std_err ** 2) + (dlag_dmu ** 2) * (mu_std_err ** 2)
            lag_time_std_err = np.sqrt(max(lag_var, 0.0))

    return lag_time, lag_time_std_err
