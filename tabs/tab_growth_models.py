import streamlit as st
import os
from PIL import Image
import numpy as np

def display_tab_growth_models():
    """
    Tab 7: Growth Models. 
    Shows textual and LaTeX descriptions of your standard models (exponential, logistic, etc.).
    """
    st.title("Bacterial Growth Models")

    image_path = os.path.join("assets","f1.png")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Example: Growth Model Visualization", width=600)
    else:
        st.warning("Image file 'f1.png' not found in the 'assets' folder.")

    st.write("## Exponential Growth Model")
    st.write("The exponential growth model is described by the following equations:")
    st.latex(r'\frac{dX}{dt} = \mu X')
    st.latex(r'x(t) = X_0 e^{\mu t}')
    st.code("""
def exponential_growth(t, mu, X0):
    return X0 * np.exp(mu * t)
""")
    st.write("where x0 is the initial bacterial biomass at time 0, mu is the growth rate")

    st.write("## Logistic Growth Model")
    st.write("The logistic growth model with saturation is described by the following equations:")
    st.latex(r'\frac{dX}{dt} = \mu X \left(1 - \frac{X}{K}\right)')
    st.latex(r'x(t) = \frac{X_0 e^{\mu t}}{1 + \frac{X_0}{K} \left(e^{\mu t} - 1\right)}')
    st.code("""
def logistic_growth(t, mu, X0, K):
    return (X0 * np.exp(mu * t)) / (1 + (X0 / K) * (np.exp(mu * t) - 1))
""")

    st.write("## Baranyi Model")
    st.write("The Baranyi model for lag-exponential growth is described by the following equations:")
    st.latex(r'\frac{dX}{dt} = \mu \frac{q(t)}{1 + q(t)} X')
    st.latex(r'\frac{dq}{dt} = \mu q')
    st.latex(r'x(t) = X_0 \frac{1 + q_0 e^{\mu t}}{1 + q_0}')
    st.code("""
def baranyi_growth(t, X0, mu, q0):
    q_t = q0 * np.exp(mu * t)
    return X0 * (1 + q_t) / (1 + q0)
""")
    st.write("where x0 is the initial biomass at time 0, mu is the growth rate, q0 is a physiological state...")

    st.write("## Lag-Exponential-Saturation Growth Model")
    st.write("The lag-exponential-saturation growth model is described by the following equations:")
    st.latex(r'\frac{dX}{dt} = \mu \frac{q(t)}{1 + q(t)} X \left(1 - \frac{X}{K}\right)')
    st.latex(r'\frac{dq}{dt} = \mu q')
    st.latex(r'x(t) = X_0 \frac{1 + q_0 e^{\mu t}}{1 + q_0 - q_0 \frac{X_0}{K} + \frac{q_0 X_0}{K} e^{\mu t}}')
    st.code("""
def lag_exponential_saturation_growth(t, mu, X0, q0, K):
    return X0 * (1 + q0 * np.exp(mu * t)) / (1 + q0 - q0 * (X0 / K) + (q0 * X0 / K) * np.exp(mu * t))
""")
    st.write("where x0 is the initial biomass at time 0, etc...")

    st.write("## Gompertz Growth Model")
    st.write("The Gompertz growth model is a sigmoid function used to describe bacterial growth. It is defined by the following differential equation:")
    st.latex(r'\frac{dX}{dt} = -B \cdot X \cdot \ln\left(\frac{X}{A}\right)')
    st.write("The solution to this differential equation is:")
    st.latex(r'X(t) = A \cdot e^{-e^{B \cdot (C - t)}}')
    st.code("""
def gompertz_growth(t, A, B, C):
    return A * np.exp(-np.exp(B * (C - t)))
""")
    st.write("where:")
    st.write("- **A**: The asymptote (maximum value).")
    st.write("- **B**: The growth rate.")
    st.write("- **C**: The time at the inflection point.")
    st.write("- **t**: Time.")

    st.write("## Power Law")
    st.write("The Power Law model is commonly used for fitting background wells:")
    st.latex(r'y(t) = a \cdot t^n + b')
    st.code("""
def power_law(t, a, n, b):
    return a * np.power(t, n) + b
""")
    st.write("where:")
    st.write("- **a**: Scaling coefficient")
    st.write("- **n**: Power exponent")
    st.write("- **b**: Y-intercept/offset")
    st.write("- **t**: Time")
    st.write("This model is particularly useful for blank well fitting where background signal follows a power relationship with time.")

    st.write("## Polynomial Function (Quadratic)")
    st.write("A simple quadratic polynomial function used for blank well fitting:")
    st.latex(r'y(t) = a \cdot t^2 + b \cdot t + c')
    st.code("""
def polynomial_func(t, a, b, c):
    return a * t**2 + b * t + c
""")
    st.write("where:")
    st.write("- **a**: Coefficient of t²")
    st.write("- **b**: Coefficient of t")
    st.write("- **c**: Constant term/Y-intercept")
    st.write("- **t**: Time")
    st.write("This model is suitable when the background signal follows a parabolic trend over time.")

    # Add this after your existing model explanations

    st.write("## Parameter Estimation Methods")
    st.write("""
### How Initial Parameter Values are Calculated

For bacterial growth models, choosing appropriate initial values for parameters is critical for successful curve fitting.
Our application uses biologically informed methods to estimate starting values:

| Parameter | Description | Initial Value Calculation |
|-----------|-------------|---------------------------|
| **μ (mu)** | Growth rate | Calculated from log-transformed OD: `(ln(OD₂) - ln(OD₁))/(t₂ - t₁)` using early time points |
| **X₀** | Initial population | Average of first 3 OD readings to smooth measurement noise |
| **K** | Carrying capacity | Maximum OD value observed in the selected time interval |
| **q₀** | Initial physiological state | Default set to 1.0 (Baranyi model) |

These methods provide the optimization algorithm with biologically realistic starting points, 
improving convergence and the likelihood of finding the global optimum rather than local minima.
""")

    st.write("### Example: Growth Rate Calculation")
    st.latex(r"\mu = \frac{\ln(OD_2) - \ln(OD_1)}{t_2 - t_1}")
    st.write("This formula represents the slope of the log-transformed growth curve during exponential phase.")

    st.write("### Why These Estimates Matter")
    st.write("""
- **Better Convergence**: Starting near the true parameter values helps curve fitting algorithms converge faster
- **Avoid Local Minima**: Realistic starting values reduce the risk of finding suboptimal solutions
- **Biological Relevance**: Parameters derived this way have direct connection to the biological process
""")

    st.write("### Parameter Interpretations")
    st.write("""
| Model | Key Parameters | Biological Interpretation |
|-------|---------------|---------------------------|
| **Exponential** | μ (mu) | Cell division rate during unconstrained growth |
| **Logistic** | K | Maximum population density (carrying capacity) |
| **Baranyi** | q₀ | Reflects lag phase duration (adaptation time) |
| **Gompertz** | C | Time at inflection point (maximum growth rate) |
""")

    st.info("""
💡 **Expert Tip**: When manually adjusting parameters, consider that growth rates (μ) for bacteria 
typically range from 0.1 to 2.0 h⁻¹, depending on the species and conditions.
""")
