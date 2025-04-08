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
