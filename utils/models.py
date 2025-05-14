import numpy as np

def power_law(x, a, n, b):
    """
    Power Law model: y = a*x^n + b
    :param x: Independent variable (time)
    :param a: Coefficient
    :param n: Power exponent
    :param b: Y-intercept/offset
    :return: y value
    """
    return a*np.power(x,n)+b

def polynomial_func(t, a, b, c):
    return a*t**2 + b*t + c

def exponential_growth(t, mu, X0):
    return X0*np.exp(mu*t)

def logistic_growth(t, mu, X0, K):
    return (X0*np.exp(mu*t)) / (1+(X0/K)*(np.exp(mu*t)-1))

def baranyi_growth(t, X0, mu, q0):
    q_t=q0*np.exp(mu*t)
    return X0*(1+q_t)/(1+q0)

def lag_exponential_saturation_growth(t, mu, X0, q0, K):
    return X0*(1+q0*np.exp(mu*t)) / (1+q0 - q0*(X0/K)+(q0*X0/K)*np.exp(mu*t))

def gompertz_growth(t, A, B, C):
    """
    Gompertz model equation:
    y(t) = A * exp(-exp(B * (C - t)))
    :param t: Time
    :param A: Asymptote (maximum value)
    :param B: Growth rate
    :param C: Time at the inflection point
    :return: Growth value at time t
    """
    return A * np.exp(-np.exp(B * (C - t)))

# Add or update this dictionary with smart defaults for ALL models:
default_guesses = {
    "Exponential Growth": lambda y_data, time_vals: [
        np.min(y_data),  # X0 (initial population)
        np.log(y_data[-1] / max(y_data[0], 0.01)) / (time_vals[-1] - time_vals[0])  # mu (growth rate)
    ],
    "Logistic Growth": lambda y_data, time_vals: [
        np.min(y_data),  # X0 (initial population) 
        1.1 * np.max(y_data),  # K (carrying capacity)
        np.log(y_data[-1] / max(y_data[0], 0.01)) / (time_vals[-1] - time_vals[0])  # r (growth rate)
    ],
    "Baranyi Growth": lambda y_data, time_vals: [
        np.min(y_data),  # y0 (initial population)
        1.2 * np.max(y_data),  # ymax (max population)
        np.log(y_data[-1] / max(y_data[0], 0.01)) / (time_vals[-1] - time_vals[0]),  # mu_max (max growth rate)
        0.1 * (time_vals[-1] - time_vals[0])  # lag (lag time - 10% of total time)
    ],
    "Lag-Exponential-Saturation Growth": lambda y_data, time_vals: [
        np.min(y_data),  # y0
        1.2 * np.max(y_data),  # ymax
        np.log(y_data[-1] / max(y_data[0], 0.01)) / (time_vals[-1] - time_vals[0]),  # mu_max
        0.2 * (time_vals[-1] - time_vals[0]),  # t_lag
        0.8 * (time_vals[-1] - time_vals[0])   # t_max
    ],
    "Gompertz Growth": lambda y_data, time_vals: [
        np.min(y_data),  # A (lower asymptote)
        np.max(y_data) - np.min(y_data),  # C (upper - lower asymptote)
        np.log(y_data[-1] / max(y_data[0], 0.01)) / (time_vals[-1] - time_vals[0]),  # B (growth rate)
        0.3 * (time_vals[-1] - time_vals[0])   # M (time at max growth)
    ]
    # Add other models as needed
}

MODEL_PARAMS = {
    "Power Law":["a","n","b"],  # Changed from "Polynomial Growth"
    "Polynomial Function":["a","b","c"],
    "Exponential Growth": ["X0", "mu"],
    "Logistic Growth": ["X0", "K", "r"],
    "Baranyi Growth": ["y0", "ymax", "mu_max", "lag"],
    "Lag-Exponential-Saturation Growth": ["y0", "ymax", "mu_max", "t_lag", "t_max"],
    "Gompertz Growth": ["A", "C", "B", "M"],
    "Custom Function":[],
    "Automatic Fit":[]
}

MODEL_FUNCTIONS={
    "Power Law":power_law,  # Changed from "Polynomial Growth"
    "Exponential Growth":exponential_growth,
    "Logistic Growth":logistic_growth,
    "Baranyi Growth":baranyi_growth,
    "Lag-Exponential-Saturation Growth":lag_exponential_saturation_growth,
    "Gompertz Growth": gompertz_growth,
    "Custom Function":None,
    "Automatic Fit":None
}

PARAMETER_UNITS={
    "mu":"[1/time]",
    "X0":"[OD]",
    "K":"[OD]"
}
