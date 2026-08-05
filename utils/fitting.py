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
