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

# Helpers for the initial guesses below. Background-subtracted data is clipped
# at zero, so a fit interval in the lag phase can be all zeros. Guessing from
# those values directly gives log(0) = -inf for the growth rate and X0 = 0,
# which makes the growth models identically zero and leaves curve_fit with
# nothing to work from. These helpers use only the real (positive) measurements.

def _positive_values(y_data):
    """Finite values above zero, in their original order."""
    y = np.asarray(y_data, dtype=float)
    return y[np.isfinite(y) & (y > 0)]


def _mu_guess(y_data, time_vals):
    """Growth rate estimated from the first and last positive measurements."""
    positives = _positive_values(y_data)
    span = float(time_vals[-1]) - float(time_vals[0])
    if len(positives) < 2 or span <= 0:
        return 1.0
    mu = np.log(positives[-1] / positives[0]) / span
    return float(mu) if np.isfinite(mu) else 1.0


def _x0_guess(y_data):
    """Smallest positive measurement; zero would flatten the model to zero."""
    positives = _positive_values(y_data)
    return float(positives.min()) if len(positives) else 0.01


def _scaled_max(y_data, factor):
    """Carrying-capacity style guess, kept positive for all-zero intervals."""
    positives = _positive_values(y_data)
    return float(factor * positives.max()) if len(positives) else 1.0


# Update these lambdas to return the CORRECT number of parameters for each function:
default_guesses = {
    "Exponential Growth": lambda y_data, time_vals: [
        _mu_guess(y_data, time_vals),  # mu (growth rate)
        _x0_guess(y_data)  # X0 (initial population)
    ],
    "Logistic Growth": lambda y_data, time_vals: [
        _mu_guess(y_data, time_vals),  # mu
        _x0_guess(y_data),  # X0
        _scaled_max(y_data, 1.1)  # K (carrying capacity)
    ],
    "Baranyi Growth": lambda y_data, time_vals: [
        _x0_guess(y_data),  # X0
        _mu_guess(y_data, time_vals),  # mu
        0.1  # q0 - simplified
    ],
    "Lag-Exponential-Saturation Growth": lambda y_data, time_vals: [
        _mu_guess(y_data, time_vals),  # mu
        _x0_guess(y_data),  # X0
        0.1,  # q0
        _scaled_max(y_data, 1.2)  # K
    ],
    "Gompertz Growth": lambda y_data, time_vals: [
        _scaled_max(y_data, 1.2),  # A (asymptote)
        0.5,  # B (growth rate coefficient)
        0.3 * (time_vals[-1] - time_vals[0])  # C (inflection point time)
    ],
    "Power Law": lambda y_data, time_vals: [
        0.1,  # a
        0.5,  # n
        _x0_guess(y_data)  # b
    ]
}

MODEL_PARAMS = {
    "Power Law": ["a", "n", "b"],
    "Polynomial Function": ["a", "b", "c"],
    "Exponential Growth": ["mu", "X0"],
    "Logistic Growth": ["mu", "X0", "K"],
    "Baranyi Growth": ["X0", "mu", "q0"],
    "Lag-Exponential-Saturation Growth": ["mu", "X0", "q0", "K"],
    "Gompertz Growth": ["A", "B", "C"],
    "Custom Function": [],
    "Automatic Fit": []
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

PARAMETER_UNITS = {
    "mu": "[1/time]",
    "X0": "[OD]",
    "K": "[OD]",
    "q0": "[dimensionless]",
    "lag_time": "[time]",
    "A": "[OD]",
    "B": "[1/time]",
    "C": "[time]",
    "a": "[OD/time^n]",
    "n": "[dimensionless]",
    "b": "[OD]"
}
