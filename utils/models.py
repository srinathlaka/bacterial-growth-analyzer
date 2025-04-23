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

# For lines referencing "MODEL_PARAMS" "MODEL_FUNCTIONS" "default_guesses" etc.:
default_guesses = {
    "Exponential Growth": lambda data, times: [
        # mu: growth rate from log-transformed OD values
        (np.log(data[min(3, len(data)-1)]) - np.log(data[0])) / (times[min(3, len(data)-1)] - times[0]) 
            if len(data) >= 2 else 0.2,
        # X0: average of first 3 OD readings
        np.mean(data[:min(3, len(data))]) if len(data) > 0 else 0.01
    ],
    "Logistic Growth": lambda data, times: [
        # mu: growth rate from log-transformed OD values
        (np.log(data[min(3, len(data)-1)]) - np.log(data[0])) / (times[min(3, len(data)-1)] - times[0]) 
            if len(data) >= 2 else 0.2,
        # X0: average of first 3 OD readings
        np.mean(data[:min(3, len(data))]) if len(data) > 0 else 0.01,
        # K: max OD value (carrying capacity)
        np.max(data) if len(data) > 0 else 1.0
    ],
    "Baranyi Growth": lambda data, times: [
        # mu: growth rate from log-transformed OD values
        (np.log(data[min(3, len(data)-1)]) - np.log(data[0])) / (times[min(3, len(data)-1)] - times[0]) 
            if len(data) >= 2 else 0.2,
        # X0: average of first 3 OD readings
        np.mean(data[:min(3, len(data))]) if len(data) > 0 else 0.01,
        # Xmax: max OD value (carrying capacity)
        np.max(data) if len(data) > 0 else 1.0,
        # q0: initial physiological state - fixed at 1.0
        1.0
    ]
    # Add similar patterns for other growth models
}

MODEL_PARAMS={
    "Power Law":["a","n","b"],  # Changed from "Polynomial Growth"
    "Polynomial Function":["a","b","c"],
    "Exponential Growth":["mu","X0"],
    "Logistic Growth":["mu","X0","K"],
    "Baranyi Growth":["X0","mu","q0"],
    "Lag-Exponential-Saturation Growth":["mu","X0","q0","K"],
    "Gompertz Growth": ["A", "B", "C"],
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
