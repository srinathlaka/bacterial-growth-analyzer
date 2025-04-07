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
        conf_interval[i]=np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T))+residual_variance)

    lower_bound=fitted_values - t_critical*conf_interval
    upper_bound=fitted_values + t_critical*conf_interval
    return lower_bound, upper_bound
