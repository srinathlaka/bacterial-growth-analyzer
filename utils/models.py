import numpy as np

def polynomial_growth(x, a, n, b):
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

# For lines referencing "MODEL_PARAMS" "MODEL_FUNCTIONS" "default_guesses" etc.:
default_guesses={
    "Exponential Growth":[0.1,1.0],
    "Logistic Growth":[0.1,1.0,2.0],
    "Baranyi Growth":[1.0,0.1,0.1],
    "Lag-Exponential-Saturation Growth":[0.1,1.0,0.1,2.0]
}

MODEL_PARAMS={
    "Polynomial Growth":["a","n","b"],
    "Polynomial Function":["a","b","c"],
    "Exponential Growth":["mu","X0"],
    "Logistic Growth":["mu","X0","K"],
    "Baranyi Growth":["X0","mu","q0"],
    "Lag-Exponential-Saturation Growth":["mu","X0","q0","K"],
    "Custom Function":[],
    "Automatic Fit":[]
}

MODEL_FUNCTIONS={
    "Exponential Growth":exponential_growth,
    "Logistic Growth":logistic_growth,
    "Baranyi Growth":baranyi_growth,
    "Lag-Exponential-Saturation Growth":lag_exponential_saturation_growth,
    "Custom Function":None,
    "Automatic Fit":None
}

PARAMETER_UNITS={
    "mu":"[1/time]",
    "X0":"[OD]",
    "K":"[OD]"
}
