import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.stats import t as t_dist
import sympy as sp
from scipy.optimize import minimize
import uuid
from PIL import Image
import ruptures as rpt
import json
from scipy.signal import savgol_filter


def detect_phases(
    time,
    od_values,
    method="thresholded_derivative",
    # --- Parameters for thresholded derivative ---
    smoothing_window=5,
    polyorder=2,
    derivative_threshold=0.005,
    min_distance=5,
    # --- Parameters for slope-based ---
    slope_window=5,
    slope_threshold=0.001
):
    """
    Detect phases in data using either a thresholded derivative approach or a slope-based approach.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values.
    od_values : np.ndarray
        1D array of OD (or other measurement) values.
    method : str, optional
        The detection method: "thresholded_derivative" or "slope_based".
    smoothing_window : int, optional
        For thresholded derivative: Savitzky-Golay smoothing window (must be odd).
    polyorder : int, optional
        For thresholded derivative: polynomial order for Savitzky-Golay filter.
    derivative_threshold : float, optional
        For thresholded derivative: minimum absolute derivative needed to register a sign change.
    min_distance : int, optional
        For thresholded derivative: minimum number of points between breakpoints.
    slope_window : int, optional
        For slope-based detection: number of points in each local window for slope calculation.
    slope_threshold : float, optional
        For slope-based detection: absolute slope cutoff to decide “high-slope” vs. “low-slope”.

    Returns
    -------
    phases : list of (float, float)
        List of (start_time, end_time) tuples for each detected phase.
    breakpoints : list of int
        List of indices in `time` where phase boundaries occur.
    extra : dict
        Additional outputs for plotting or debugging. For "thresholded_derivative",
        this includes the derivative array. For "slope_based", this includes the slopes array.
    """
    phases = []
    breakpoints = []
    extra = {}

    if method == "thresholded_derivative":
        # Ensure the smoothing window is odd
        if smoothing_window % 2 == 0:
            smoothing_window += 1

        # 1. Smooth data
        try:
            smoothed = savgol_filter(od_values, window_length=smoothing_window, polyorder=polyorder)
        except ValueError:
            st.error("Savgol filter error: window_length might be too large for the data length.")
            return [], [], {}
        # 2. Compute derivative
        derivative = np.gradient(smoothed, time)
        extra["derivative"] = derivative

        # 3. Identify sign changes above the threshold
        sign = np.sign(derivative)
        sign_changes = []
        for i in range(1, len(sign)):
            if sign[i] != sign[i - 1]:
                if abs(derivative[i]) > derivative_threshold or abs(derivative[i - 1]) > derivative_threshold:
                    sign_changes.append(i)

        # 4. Enforce minimum distance
        filtered_changes = []
        last_cp = -min_distance
        for cp in sign_changes:
            if cp - last_cp >= min_distance:
                filtered_changes.append(cp)
                last_cp = cp
        breakpoints = filtered_changes

        # 5. Convert breakpoints to (start, end) intervals
        start_idx = 0
        for bp in breakpoints:
            end_idx = bp
            if end_idx == 0:
                continue
            phases.append((time[start_idx], time[end_idx - 1]))
            start_idx = bp
        if start_idx < len(time):
            phases.append((time[start_idx], time[-1]))

    elif method == "slope_based":
        # 1. Calculate local slopes in a sliding window
        slopes = []
        for i in range(len(time) - slope_window):
            y_segment = od_values[i : i + slope_window]
            x_segment = time[i : i + slope_window]
            A = np.vstack([x_segment, np.ones(len(x_segment))]).T
            m, _ = np.linalg.lstsq(A, y_segment, rcond=None)[0]
            slopes.append(m)
        # Pad out to match length
        slopes = np.array(slopes + [slopes[-1]] * slope_window)
        extra["slopes"] = slopes

        # 2. Identify slope changes
        high_slope = np.abs(slopes) > slope_threshold
        slope_changes = []
        for i in range(1, len(high_slope)):
            if high_slope[i] != high_slope[i - 1]:
                slope_changes.append(i)
        breakpoints = slope_changes

        # 3. Convert breakpoints to (start, end) intervals
        start_idx = 0
        for bp in breakpoints:
            phases.append((time[start_idx], time[bp - 1]))
            start_idx = bp
        if start_idx < len(time):
            phases.append((time[start_idx], time[-1]))

    else:
        st.error("Unknown method. Use 'thresholded_derivative' or 'slope_based'.")
        return [], [], {}

    return phases, breakpoints, extra


def plot_detected_phases(time, od_values, phases, 
                         derivative=None, slopes=None, 
                         change_points=None):
    """
    Plot the OD values with detected phases highlighted.
    Optionally plot the derivative or slopes.
    """
    fig = go.Figure()

    # Original OD curve
    fig.add_trace(go.Scatter(
        x=time,
        y=od_values,
        mode='lines+markers',
        name='OD Values',
        line=dict(color='blue')
    ))

    # Plot derivative if given
    if derivative is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=derivative,
            mode='lines',
            name='Derivative',
            line=dict(color='orange', dash='dash')
        ))

    # Plot slopes if given
    if slopes is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=slopes,
            mode='lines',
            name='Slopes',
            line=dict(color='purple', dash='dot')
        ))

    # Highlight each phase
    colors = ['rgba(255, 0, 0, 0.2)',
              'rgba(0, 255, 0, 0.2)',
              'rgba(0, 0, 255, 0.2)',
              'rgba(255, 255, 0, 0.2)',
              'rgba(255, 165, 0, 0.2)']
    for idx, (start, end) in enumerate(phases):
        color = colors[idx % len(colors)]
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0,
            annotation_text=f"Phase {idx + 1}",
            annotation_position="top left"
        )

    # Mark change points
    if change_points is not None:
        for cp in change_points:
            if cp < len(time):
                fig.add_vline(
                    x=time[cp],
                    line=dict(color='red', dash='dot'),
                    annotation_text="Change Point",
                    annotation_position="top left"
                )

    fig.update_layout(
        title='Automatic Phase Detection',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
def plot_all_ode_fits_summary(ode_fits, operated_data, selected_operated_wells):
    """
    Create a summary plot that shows each fit's fitted ODE solutions for X and Y
    along with the overall average data.
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()

    # Plot overall average data
    fig.add_trace(go.Scatter(
        x=operated_data["Time"],
        y=operated_data["Average"],
        mode='lines',
        name='Overall Average',
        line=dict(color='black', width=2, dash='dot')
    ))

    # Loop over each fit
    for i, fit_item in enumerate(ode_fits):
        fit_results = fit_item.get("fit_results")
        if not fit_results:
            continue

        # Adjust keys to match how your code stores them; e.g. "fit_time" or "phase_time"
        fit_t = fit_results.get("fit_time")      # was "phase_time" in original
        sol_y = fit_results.get("solution_y")    # 2D array, shape = (n_states, n_time)

        if fit_t is None or sol_y is None:
            continue

        # Plot the X solution (first row of sol_y)
        fig.add_trace(go.Scatter(
            x=fit_t,
            y=sol_y[0],
            mode='lines',
            name=f"Fit {i+1} X",
            line=dict(width=2, color='green')
        ))

        # Plot the Y solution (second row), if available
        if sol_y.shape[0] > 1:
            fig.add_trace(go.Scatter(
                x=fit_t,
                y=sol_y[1],
                mode='lines',
                name=f"Fit {i+1} Y",
                line=dict(width=2, color='red', dash='dash')
            ))
        
        # Plot CI for X if available
        lower = fit_results.get("lower_bound")
        upper = fit_results.get("upper_bound")
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([fit_t, fit_t[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(173,216,230,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name=f"Fit {i+1} X 95% CI",
                showlegend=False
            ))
        
        # Annotate with fitted parameter values (if present)
        params = fit_results.get("parameters")
        # If you stored the param names at fit_item["parameters"], adjust accordingly
        param_names = fit_item.get("parameters")
        if params is not None and param_names is not None:
            annotation_text = ", ".join([f"{name}={val:.3f}" for name, val in zip(param_names, params)])
            t_mid = fit_t[len(fit_t)//2]
            y_mid = np.max(sol_y[0])
            fig.add_annotation(
                x=t_mid,
                y=y_mid,
                text=annotation_text,
                showarrow=True,
                arrowhead=1
            )

    fig.update_layout(
        title="Summary: All ODE Fits (X & Y)",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_white"
    )
    return fig






def compute_ode_ci_for_X(ode_system_func, param_values, param_cov, y0, t_eval, 
                         alpha=0.05, solver_method="RK45", eps=1e-6):
    """
    Compute confidence intervals for the X variable (first state) of an ODE solution.
    
    Parameters:
      ode_system_func : callable
          A function of the form f(t, y, param_values) returning derivatives.
      param_values : array-like
          Best-fit parameters (numeric).
      param_cov : 2D array
          Covariance matrix for the parameters.
      y0 : array-like
          Initial conditions.
      t_eval : array-like
          Time points at which the ODE is evaluated.
      alpha : float, optional
          Significance level (default 0.05 for 95% CI).
      solver_method : str, optional
          Method for solve_ivp.
      eps : float, optional
          Finite-difference step size.
    
    Returns:
      dict with keys:
         "t": time points,
         "nominal_X": nominal ODE solution for X,
         "CI_lower": lower bound for X,
         "CI_upper": upper bound for X,
         "std": standard error for X at each time.
    """
    # Solve nominal ODE
    sol_nom = solve_ivp(
        fun=lambda t, y: ode_system_func(t, y, param_values),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method=solver_method
    )
    if not sol_nom.success:
        st.error("ODE solver failed: " + sol_nom.message)
        return None
    # Assume X is the first variable
    nominal_X = sol_nom.y[0]  # shape (len(t_eval),)
    n_time = len(t_eval)
    n_params = len(param_values)
    
    # Compute finite-difference gradients for X with respect to each parameter
    grad_X = np.zeros((n_time, n_params))
    for j in range(n_params):
        dp = np.zeros_like(param_values)
        dp[j] = eps
        sol_pert = solve_ivp(
            fun=lambda t, y: ode_system_func(t, y, param_values + dp),
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method=solver_method
        )
        if not sol_pert.success:
            grad_X[:, j] = 0.0
        else:
            grad_X[:, j] = (sol_pert.y[0] - nominal_X) / eps
    
    # Propagate parameter covariance to get variance for X at each time point
    std_X = np.zeros(n_time)
    for i in range(n_time):
        g = grad_X[i, :]  # gradient at time t_i
        var_i = g @ param_cov @ g.T
        std_X[i] = np.sqrt(max(var_i, 0))
    
    # Use a critical value (normal approx, ~1.96 for 95% CI)
    crit_val = t_dist.ppf(1 - alpha/2, df=np.inf)
    CI_lower = nominal_X - crit_val * std_X
    CI_upper = nominal_X + crit_val * std_X
    
    return {
        "t": sol_nom.t,
        "nominal_X": nominal_X,
        "CI_lower": CI_lower,
        "CI_upper": CI_upper,
        "std": std_X
    }



#display ode equations
def display_ode_equations(variables, odes):
    """
    Generate and display ODE equations in LaTeX format.

    Parameters:
    - variables (list): List of variable names.
    - odes (list): List of ODE expressions.
    """
    st.write("### ODE Equations")
    for var, ode in zip(variables, odes):
        try:
            # Use SymPy to parse the ODE expression and convert to LaTeX
            ode_expr = sp.sympify(ode)
            equation = f"\\frac{{d{var}}}{{dt}} = {sp.latex(ode_expr)}"
            st.latex(equation)
        except Exception as e:
            st.error(f"Error displaying equation for variable '{var}': {e}")
#ode fitting
def parse_ode(ode_expr, variables, parameters):
    """
    Parse the ODE expression into a callable function.
    Example: "r * X" becomes a function that computes r * X.
    """
    try:
        # Define symbols
        t_sym, *var_syms = sp.symbols(['t'] + variables)
        param_syms = sp.symbols(parameters)

        # Create a dictionary for sympy
        local_dict = {var: var_sym for var, var_sym in zip(variables, var_syms)}
        local_dict.update({param: param_sym for param, param_sym in zip(parameters, param_syms)})
        local_dict['t'] = t_sym

        # Parse the expression
        expr = sp.sympify(ode_expr, locals=local_dict)

        # Convert to a lambda function
        func = sp.lambdify((t_sym, var_syms, param_syms), expr, 'numpy')

        def ode_func(t, y, params):
            # y is a list/array in the order of 'variables'
            return func(t, y, params)

        return ode_func
    except Exception as e:
        st.error(f"Error parsing ODE expression '{ode_expr}': {e}")
        return None


def solve_custom_ode(ode_funcs, y0, params, t_span, t_eval, variables, observed_data=None):
    """
    Solve a system of custom ODEs with an option to dynamically adjust initial conditions.

    Parameters:
    - ode_funcs (list): List of callable ODE functions.
    - y0 (list): Initial conditions for the variables.
    - params (list): Parameters for the ODEs.
    - t_span (tuple): Time span for the solution.
    - t_eval (array): Time points at which to store the solution.
    - variables (list): List of variable names.
    - observed_data (dict): A dictionary containing variable names as keys and observed data arrays as values.
                            The initial condition of the first variable will be adjusted to the starting value
                            of its observed data within the time range.

    Returns:
    - sol (OdeResult): The solution object from solve_ivp.
    """
    def system(t, y):
        dydt = []
        for idx, func in enumerate(ode_funcs):
            try:
                dydt_value = func(t, y, params)
                dydt.append(dydt_value)
            except Exception as e:
                st.error(f"Error evaluating ODE for variable '{variables[idx]}' at time {t:.2f}: {e}")
                raise
        return dydt

    try:
        # Adjust initial condition of the first variable based on observed data
        if observed_data is not None and len(y0) > 0 and variables[0] in observed_data:
            observed_values = observed_data[variables[0]]
            observed_time = observed_data["Time"]
            mask = (observed_time >= t_span[0]) & (observed_time <= t_span[1])
            if mask.any():
                y0[0] = observed_values[mask].iloc[0]  # Dynamically set initial condition for the first variable

        # Solve the system of ODEs
        sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')
        if not sol.success:
            st.error(f"ODE Solver failed: {sol.message}")
            return None
        return sol
    except Exception as e:
        st.error(f"Error solving ODEs: {e}")
        return None

def plot_ode_solution(sol, variables, title="ODE Solutions"):
    """
    Plot the ODE solutions using Plotly.

    Parameters:
    - sol (OdeResult): The solution object from solve_ivp.
    - variables (list): List of variable names.
    - title (str): Title of the plot.

    Returns:
    - fig (Plotly Figure): The generated plot.
    """
    fig = go.Figure()
    for idx, var in enumerate(variables):
        fig.add_trace(go.Scatter(
            x=sol.t,
            y=sol.y[idx],
            mode='lines',
            name=var
        ))
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    return fig

def validate_phase_inputs(variables, parameters, odes):
    """Validate the inputs for the ODE phase."""
    if not variables:
        st.error("At least one variable is required.")
        return False
    if not parameters:
        st.error("At least one parameter is required.")
        return False
    if len(odes) != len(variables):
        st.error("Number of ODE expressions must match number of variables.")
        return False
    if any(not ode.strip() for ode in odes):
        st.error("All ODE expressions must be non-empty.")
        return False
    return True

#D efine growth models
def polynomial_growth(x, a, n, b):
    return a * np.power(x, n) + b


def polynomial_func(t, a, b, c):
    return a * t**2 + b * t + c


def exponential_growth(t, mu, X0):
    return X0 * np.exp(mu * t)


def logistic_growth(t, mu, X0, K):
    return (X0 * np.exp(mu * t)) / (1 + (X0 / K) * (np.exp(mu * t) - 1))


def baranyi_growth(t, X0, mu, q0):
    q_t = q0 * np.exp(mu * t)
    return X0 * (1 + q_t) / (1 + q0)


def lag_exponential_saturation_growth(t, mu, X0, q0, K):
    return X0 * (1 + q0 * np.exp(mu * t)) / (1 + q0 - q0 * (X0 / K) + (q0 * X0 / K) * np.exp(mu * t))


# Function to calculate model metrics
def calculate_metrics(observed, predicted, num_params):
    residuals = observed - predicted
    rss = np.sum(residuals**2)
    r_squared = 1 - (rss / np.sum((observed - np.mean(observed))**2))
    aic = 2 * num_params + len(observed) * np.log(rss / len(observed))
    return rss, r_squared, aic


# Compute normal confidence intervals
from scipy.optimize import approx_fprime
from scipy.stats import t
import numpy as np

from scipy.optimize import approx_fprime
from scipy.stats import t
import numpy as np

def compute_confidence_intervals(time, params, covariance, alpha, dof, residual_variance, model_func):
    """
    Compute confidence intervals for fitted data.

    Parameters:
    - time: Array of time values.
    - params: Fitted model parameters.
    - covariance: Covariance matrix of the fitted parameters.
    - alpha: Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    - dof: Degrees of freedom for the fit.
    - residual_variance: Variance of the residuals.
    - model_func: The growth model function.

    Returns:
    - lower_bound: Lower confidence interval values.
    - upper_bound: Upper confidence interval values.
    """
    t_critical = t.ppf(1 - alpha / 2, dof)
    epsilon = np.sqrt(np.finfo(float).eps)  # Small value for numerical approximation
    conf_interval = np.zeros(len(time))
    fitted_values = model_func(time, *params)

    for i in range(len(time)):
        def func(p):
            return model_func(np.array([time[i]]), *p)

        # Approximate gradient (Jacobian row)
        gradient = approx_fprime(params, func, epsilon)

        # Confidence interval computation
        conf_interval[i] = np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T)) + residual_variance)


    lower_bound = fitted_values - t_critical * conf_interval
    upper_bound = fitted_values + t_critical * conf_interval

    return lower_bound, upper_bound



# Select layout for wells
def select_layout():
    layout_option = st.selectbox("Select layout option", ["Select from presets", "Custom"])
    if layout_option == "Select from presets":
        well_format = st.selectbox("Select well format", ["96 well rows 8 column 12", "24 well rows 4 column 6", 
                                                         "84 well rows 7 column 12", "1536 well rows 32 column 48"])
        if well_format == "24 well rows 4 column 6":
            return 4, 6
        elif well_format == "96 well rows 8 column 12":
            return 8, 12
        elif well_format == "84 well rows 7 column 12":
            return 7, 12
        elif well_format == "1536 well rows 32 column 48":
            return 32, 48
    else:
        custom_rows = st.number_input("Enter number of rows", min_value=1, step=1)
        custom_columns = st.number_input("Enter number of columns", min_value=1, step=1)
        return int(custom_rows), int(custom_columns)

# Perform operations between two groups' background-subtracted data
# Function to perform operations on background-subtracted data between two groups
# Perform operations on background-subtracted data between two groups with dynamic labeling
def perform_group_operations(group1_data, group2_data, operation):
    """
    Perform operations (Add, Subtract, Multiply, Divide) between two groups of background-subtracted data.

    Parameters:
    - group1_data: DataFrame for Group 1 (contains 'Time' and sample wells).
    - group2_data: DataFrame for Group 2 (contains 'Time' and sample wells).
    - operation: String specifying the operation ('Add', 'Subtract', 'Multiply', 'Divide').

    Returns:
    - operated_data: DataFrame containing the result of the operation and the calculated Average column.
    """
    # Initialize a new DataFrame to store the operated results
    operated_data = pd.DataFrame({"Time": group1_data["Time"]})

    # Get the sample wells for each group
    sample_wells1 = list(group1_data.columns[1:])  # Exclude "Time" column
    sample_wells2 = list(group2_data.columns[1:])

    # Ensure equal length of sample wells for pairing
    min_len = min(len(sample_wells1), len(sample_wells2))
    sample_wells1 = sample_wells1[:min_len]
    sample_wells2 = sample_wells2[:min_len]

    # Define the symbol to use for each operation
    symbol_map = {
        "Add": "+",
        "Subtract": "-",
        "Multiply": "*",
        "Divide": "/"
    }
    symbol = symbol_map.get(operation, "+")  # Default to "+" if operation is unknown

    # Perform the operation on each pair of sample wells and set dynamic labels
    for well1, well2 in zip(sample_wells1, sample_wells2):
        new_label = f"{well1} {symbol} {well2}"  # Create a dynamic label based on operation
        if operation == "Add":
            operated_data[new_label] = group1_data[well1] + group2_data[well2]
        elif operation == "Subtract":
            operated_data[new_label] = group1_data[well1] - group2_data[well2]
        elif operation == "Multiply":
            operated_data[new_label] = group1_data[well1] * group2_data[well2]
        elif operation == "Divide":
            operated_data[new_label] = group1_data[well1] / group2_data[well2].replace(0, np.nan)  # Avoid divide-by-zero

    # Calculate the average across all operated wells
    if operated_data.shape[1] > 1:  # Ensure there are valid operated columns
        operated_data["Average"] = operated_data.iloc[:, 1:].mean(axis=1)

    return operated_data



# Plot fitted curves with Plotly
def plot_fitted_curves(df, time, observed, fitted, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=observed, mode='lines', name='Observed Data'))
    fig.add_trace(go.Scatter(x=time, y=fitted, mode='lines', name=f'Fitted Curve ({model_name})', line=dict(color='red')))
    fig.update_layout(title=f'Fitted {model_name} Model', xaxis_title='Time', yaxis_title='OD', legend_title='Legend', template='plotly_white')
    st.plotly_chart(fig)

# Upload file
def upload_file():
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    return uploaded_file


# Read data
@st.cache_data
def read_data(uploaded_file, rows, columns):
    if uploaded_file is not None:
        try:
            st.success("File uploaded successfully!")
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, header=None)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
            df = adjust_dataframe_layout(df, rows, columns)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None


# Adjust DataFrame layout to match specified rows and columns
def adjust_dataframe_layout(df, rows, columns):
    total_columns = rows * columns
    labels = generate_labels(rows, columns)
    
    if df.shape[1] < total_columns + 1:
        extra_cols = pd.DataFrame(np.nan, index=df.index, columns=range(df.shape[1], total_columns + 1))
        df = pd.concat([df, extra_cols], axis=1)
    
    df = df.iloc[:, :total_columns + 1]
    df.columns = ["Time"] + labels
    return df


# Generate labels for well selection
def generate_labels(rows, cols):
    labels = []
    for i in range(rows):
        for j in range(cols):
            label = chr(ord('A') + i) + str(j + 1)
            labels.append(label)
    return labels

def get_button_layout_table(rows, columns):
    """
    Generate a table preview of the well layout.
    Each row shows the row letter and the concatenated well labels.
    """
    data = []
    for i in range(rows):
        row_letter = chr(ord('A') + i)
        wells = " ".join([f"{row_letter}{j+1}" for j in range(columns)])
        data.append([row_letter, wells])
    df_layout = pd.DataFrame(data, columns=["Row", "Wells"])
    return df_layout

def create_button_layout(rows, columns, labels, key_prefix, df=None):
    """
    Well selection UI with minimal gaps:
    - Top row: column radios.
    - Next row: centered "Select Wells" button.
    - Subsequent rows: each row has a row radio + row label + well buttons.
    - "Select Wells" applies intersection-based selection.
    - Manual toggling and "Clear Selection" are provided.
    """
    # Inject minimal CSS to reduce default padding/margins.
    st.markdown("""
    <style>
      .block-container {
          padding-top: 0rem;
          padding-bottom: 0rem;
      }
      .stButton, .stRadio {
          margin: 0rem;
          padding: 0rem;
      }
      div[data-testid="column"] {
          padding: 0rem 0.2rem;
      }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state if not already set.
    for k in ["selected_wells", "selected_rows", "selected_cols", "manual_deselected"]:
        if f"{key_prefix}_{k}" not in st.session_state:
            st.session_state[f"{key_prefix}_{k}"] = set()
    
    st.subheader(f"🔬 Well Selection: {key_prefix}")
    
    # Determine wells with data.
    wells_with_data = set(labels)
    if df is not None:
        wells_with_data = {well for well in labels if well in df.columns and not df[well].dropna().empty}
    
    # Optional "Select All" checkbox.
    select_all = st.checkbox("✅ Select All Wells", key=f"{key_prefix}_select_all")
    if select_all:
        st.session_state[f"{key_prefix}_selected_wells"] = set(labels)
        st.session_state[f"{key_prefix}_selected_rows"].clear()
        st.session_state[f"{key_prefix}_selected_cols"].clear()
        st.session_state[f"{key_prefix}_manual_deselected"].clear()
    
    # ─────────────────────────────────────────────
    # Top row: Column Radios.
    top_row = st.columns(columns + 1, gap="small")
    top_row[0].write(" ")  # Blank corner.
    for j in range(columns):
        col_key = f"{key_prefix}_col_select_{j}"
        top_row[j+1].radio(" ", [" ", "✓"], index=0, key=col_key, label_visibility="collapsed")
    
    # ─────────────────────────────────────────────
    # Separate row for "Select Wells" button (centered).
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    apply_selection = st.button("Select Wells", key=f"{key_prefix}_apply_selection")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # For each row: Row radio + well buttons.
    for i in range(rows):
        row_layout = st.columns(columns + 1, gap="small")
        row_key = f"{key_prefix}_row_select_{i}"
        # First cell: row radio and row label.
        row_layout[0].radio(" ", [" ", "✓"], index=0, key=row_key, label_visibility="collapsed")
        row_layout[0].write(f"**{chr(65 + i)}**")
        for j in range(columns):
            idx = i * columns + j
            well_name = labels[idx]
            is_selected = well_name in st.session_state[f"{key_prefix}_selected_wells"]
            # Show only the well name on the button (no dynamic tick here)
            btn_label = well_name
            if row_layout[j+1].button(btn_label, key=f"{key_prefix}_{well_name}"):
                # Manual toggle: update selection state.
                if is_selected:
                    st.session_state[f"{key_prefix}_selected_wells"].remove(well_name)
                    st.session_state[f"{key_prefix}_manual_deselected"].add(well_name)
                else:
                    st.session_state[f"{key_prefix}_selected_wells"].add(well_name)
                    st.session_state[f"{key_prefix}_manual_deselected"].discard(well_name)
    
    # ─────────────────────────────────────────────
    # When "Select Wells" is clicked, apply intersection logic.
    if apply_selection:
        selected_cols_1b = {j + 1 for j in range(columns)
                            if st.session_state.get(f"{key_prefix}_col_select_{j}", " ") == "✓"}
        selected_rows_1b = {i + 1 for i in range(rows)
                            if st.session_state.get(f"{key_prefix}_row_select_{i}", " ") == "✓"}
        st.session_state[f"{key_prefix}_selected_cols"] = selected_cols_1b
        st.session_state[f"{key_prefix}_selected_rows"] = selected_rows_1b
        
        max_col_1b = max(selected_cols_1b) if selected_cols_1b else -1
        max_row_1b = max(selected_rows_1b) if selected_rows_1b else -1
        
        idx = 0
        for r in range(rows):
            row_1b = r + 1
            for c in range(columns):
                col_1b = c + 1
                well_name = labels[idx]
                idx += 1
                manually_deselected = well_name in st.session_state[f"{key_prefix}_manual_deselected"]
                manually_selected = well_name in st.session_state[f"{key_prefix}_selected_wells"]
                # Intersection logic:
                if selected_rows_1b and not selected_cols_1b:
                    should_select = (row_1b in selected_rows_1b)
                elif selected_cols_1b and not selected_rows_1b:
                    should_select = (col_1b in selected_cols_1b)
                else:
                    should_select = (
                        (row_1b in selected_rows_1b and col_1b <= max_col_1b) or
                        (col_1b in selected_cols_1b and row_1b <= max_row_1b)
                    )
                if should_select and not manually_deselected:
                    st.session_state[f"{key_prefix}_selected_wells"].add(well_name)
                elif not manually_selected:
                    st.session_state[f"{key_prefix}_selected_wells"].discard(well_name)
        st.success("✅ Wells selected based on row/column intersection!")
    
    # ─────────────────────────────────────────────
    # "Clear Selection" Button.
    if st.button("Clear Selection", key=f"{key_prefix}_clear"):
        if select_all:
            st.warning("⚠️ Please uncheck 'Select All Wells' first.")
        else:
            st.session_state[f"{key_prefix}_selected_wells"].clear()
            st.session_state[f"{key_prefix}_selected_rows"].clear()
            st.session_state[f"{key_prefix}_selected_cols"].clear()
            st.session_state[f"{key_prefix}_manual_deselected"].clear()
            st.success("✅ Selection cleared!")
            st.rerun()
    
    # ─────────────────────────────────────────────
    # Display selected wells as pills below.
    selected_wells = sorted(st.session_state[f"{key_prefix}_selected_wells"])
    if selected_wells:
        st.markdown("### ✅ Selected Wells")
        st.markdown(" ".join([
            f"<span style='padding:6px 12px; margin:4px; background:#d1e7dd; border-radius:20px;'>{w}</span>"
            for w in selected_wells
        ]), unsafe_allow_html=True)
    else:
        st.info("No wells selected.")
    
    return selected_wells

# Perform background subtraction for each group with selected blank and sample wells
# Existing background subtraction function, modified to handle no blank wells case
def perform_background_subtraction(groups_data, df, group_num, selected_blank_wells, selected_sample_wells):
    if len(selected_sample_wells) == 0:
        # If no sample wells are selected, we cannot proceed
        st.warning(f"No sample wells selected for Group {group_num}. Please select at least one sample well.")
        return groups_data

    if len(selected_blank_wells) == 0:
        # No blank wells selected, subtract zero and warn the user
        st.warning(f"No blank wells selected for Group {group_num}. Proceeding without background subtraction (subtracting zero).")
        blank_mean = 0
    else:
        # Calculate the mean of the selected blank wells for background subtraction
        blank_mean = df[selected_blank_wells].mean(axis=1)

    # Create a copy of the original data for this group and perform the background subtraction (or zero subtraction)
    group_df = df.copy()
    for sample_well in selected_sample_wells:
        group_df[sample_well] = (df[sample_well] - blank_mean).clip(lower=0)

    # Store the background-subtracted data and sample wells for this group
    groups_data[f"Group_{group_num}_bg_subtracted"] = group_df[["Time"] + selected_sample_wells]
    groups_data[f"Group_{group_num}_sample_wells"] = selected_sample_wells
    return groups_data


import plotly.graph_objects as go

import numpy as np

def plot_phase_fit_with_ci(phase_fits, operated_data, selected_operated_wells):
    """
    Plot all phase fits with their confidence intervals and standard deviations on a single Plotly figure.

    Parameters:
    - phase_fits: List of dictionaries containing phase fit data.
    - operated_data: DataFrame containing the operated data.
    - selected_operated_wells: List of selected wells for plotting.

    Returns:
    - Plotly Figure.
    """
    fig = go.Figure()

    # Always calculate the average over the selected operated wells
    average_values = operated_data[selected_operated_wells].mean(axis=1)

    # Plot the overall average data
    fig.add_trace(go.Scatter(
        x=operated_data["Time"],
        y=average_values,
        mode='lines',
        name='Average Data',
        line=dict(color='black', width=2)
    ))

    # Plot each phase fit
    for fit in phase_fits:
        phase_num = fit['phase']
        model_name = fit['model']
        phase_time = fit['phase_time']
        y_pred = fit['fit']
        lower = fit['lower_bound']
        upper = fit['upper_bound']
        std_dev = fit['std_dev']

        # Confidence Interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([phase_time, phase_time[::-1]]),
            y=np.concatenate([lower, upper[::-1]]),
            fill='toself',
            fillcolor='rgba(173, 216, 230, 0.4)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            name=f'Phase {phase_num} 95% CI',
            showlegend=False
        ))

        # Fitted Data
        fig.add_trace(go.Scatter(
            x=phase_time,
            y=y_pred,
            mode='lines',
            name=f'Phase {phase_num} Fit ({model_name})',
            line=dict(width=2)
        ))

        # Standard Deviation Bands
        upper_sd = (y_pred + std_dev).tolist()
        lower_sd = (y_pred - std_dev).tolist()
        fig.add_trace(go.Scatter(
            x=phase_time.tolist() + phase_time.tolist()[::-1],
            y=upper_sd + lower_sd[::-1],
            fill='toself',
            fillcolor='rgba(144, 238, 144, 0.3)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            name=f'Phase {phase_num} Std Dev',
            showlegend=False
        ))

    fig.update_layout(
        title='All Phase Fits with Confidence Intervals and Standard Deviations',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )

    return fig


# Compute confidence intervals on the blank well fit
def compute_confidence_intervals(time, params, covariance, alpha, dof, residual_variance, model_func):
    """
    Compute confidence intervals for fitted data.

    Parameters:
    - time: Array of time values.
    - params: Fitted model parameters.
    - covariance: Covariance matrix of the fitted parameters.
    - alpha: Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    - dof: Degrees of freedom for the fit.
    - residual_variance: Variance of the residuals.
    - model_func: The growth model function.

    Returns:
    - lower_bound: Lower confidence interval values.
    - upper_bound: Upper confidence interval values.
    """
    t_critical = t.ppf(1 - alpha / 2, dof)
    epsilon = np.sqrt(np.finfo(float).eps)  # Small value for numerical approximation
    conf_interval = np.zeros(len(time))
    fitted_values = model_func(time, *params)

    for i in range(len(time)):
        # Approximate gradient (Jacobian row)
        gradient = approx_fprime(params, lambda p: model_func(time[i], *p), epsilon)

        # Confidence interval computation
        conf_interval[i] = np.sqrt(np.dot(gradient, np.dot(covariance, gradient.T)) + residual_variance)

    lower_bound = fitted_values - t_critical * conf_interval
    upper_bound = fitted_values + t_critical * conf_interval

    return lower_bound, upper_bound


def display_single_well_preview(df, well_name):
    """
    Display a small preview plot for a selected well.

    Parameters:
    - df: DataFrame containing well data.
    - well_name: The name of the well to display.
    """
    if well_name and well_name in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df[well_name], mode='lines', name=f'{well_name}'))
        
        # Customize the layout for a smaller preview plot
        fig.update_layout(
            title=f'Preview of {well_name}',
            xaxis_title='Time',
            yaxis_title='OD',
            template='plotly_white',
            width=400,  # Adjust width
            height=300,  # Adjust height
            showlegend=False
        )
        # Supply a unique key for every st.plotly_chart call:
        st.plotly_chart(fig, key=f"preview_{well_name}_{uuid.uuid4().hex}")
    else:
        st.write("No well selected for preview.")

        
# Plot confidence intervals and SD with Plotly for blank well fit
def plot_confidence_intervals(df, lower_bound, upper_bound, y_pred, std_dev):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([df['Time'], df['Time'][::-1]]),
        y=np.concatenate([lower_bound, upper_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(173, 216, 230, 0.4)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average'],
        mode='lines',
        name='Observed Data',
        line=dict(color='royalblue')
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'].tolist() + df['Time'].tolist()[::-1],
        y=(df['Average'] - std_dev).tolist() + (df['Average'] + std_dev).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(144, 238, 144, 0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Standard Deviation'
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=y_pred,
        mode='lines',
        name='Fitted Curve',
        line=dict(color='crimson')
    ))

    fig.update_layout(
        title='Confidence Intervals with Standard Deviation and Fitted Curve',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    st.plotly_chart(fig)

def plot_selected_wells(df, selected_wells):
    fig = go.Figure()
    for well in selected_wells:
        fig.add_trace(go.Scatter(x=df['Time'], y=df[well], mode='lines', name=well))
    fig.update_layout(
        title="Time vs Selected Well's OD",
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    
    # Generate a unique key using UUID
    unique_key = f"plot_selected_wells_{'_'.join(selected_wells)}_{uuid.uuid4().hex}"
    st.plotly_chart(fig, key=unique_key)

# Plot average of selected wells with Plotly
def plot_average(df, selected_wells):
    if len(selected_wells) > 0:
        selected_wells_list = list(selected_wells)
        df['Average'] = df[selected_wells_list].mean(axis=1)
        df['Std Dev'] = df[selected_wells_list].std(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Average'], mode='lines', name='Average Measurement'))

        fig.add_trace(go.Scatter(
            x=df['Time'].tolist() + df['Time'].tolist()[::-1],
            y=(df['Average'] - df['Std Dev']).tolist() + (df['Average'] + df['Std Dev']).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 80, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Standard Deviation'
        ))

        fig.update_layout(
            title='Average Measurement for Selected Wells Over Time',
            xaxis_title='Time',
            yaxis_title='Average Measurement',
            legend_title='Legend',
            template='plotly_white'
        )

        st.plotly_chart(fig)
        return 'Average', df['Average']
    
def plot_avg_sd_bg_subtracted(group_df, sample_wells, group_num):
    """
    Plot the average and standard deviation of background-subtracted data for a specified group.

    Parameters:
    - group_df: DataFrame containing background-subtracted data.
    - sample_wells: List of sample wells selected for the group.
    - group_num: Integer representing the group number (for labeling).
    """
    if len(sample_wells) > 0:
        avg_data = group_df[sample_wells].mean(axis=1)
        std_dev = group_df[sample_wells].std(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group_df['Time'], y=avg_data, mode='lines', name=f'Group {group_num} Average'))

        fig.add_trace(go.Scatter(
            x=group_df['Time'].tolist() + group_df['Time'].tolist()[::-1],
            y=(avg_data - std_dev).tolist() + (avg_data + std_dev).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(144, 238, 144, 0.3)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Standard Deviation'
        ))

        fig.update_layout(
            title=f'Average and Standard Deviation for Background-Subtracted Data (Group {group_num})',
            xaxis_title='Time',
            yaxis_title='OD',
            legend_title='Legend',
            template='plotly_white'
        )

        # Use a unique key to prevent duplicate IDs
        unique_key = f"plot_avg_sd_bg_subtracted_group_{group_num}_{uuid.uuid4().hex}"
        st.plotly_chart(fig, key=unique_key)

def plot_avg_sd_operated(data, selected_wells):
    """
    Plot average and standard deviation of operated data.

    Parameters:
    - data: DataFrame containing operated data.
    - selected_wells: List of selected wells for plotting.
    """
    avg = data["Average"]
    sd = data[selected_wells].std(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Time"],
        y=avg,
        mode='lines',
        name='Average',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data["Time"],
        y=avg + sd,
        mode='lines',
        name='Average + SD',
        line=dict(color='lightblue'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=data["Time"],
        y=avg - sd,
        mode='lines',
        name='Average - SD',
        line=dict(color='lightblue'),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.3)',
        showlegend=False
    ))
    fig.update_layout(
        title="Average and Standard Deviation of Operated Data",
        xaxis_title="Time",
        yaxis_title="OD",
        template='plotly_white'
    )

    # Generate a unique key to avoid duplicate IDs
    unique_key = f"plot_avg_sd_operated_{uuid.uuid4().hex}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

def create_custom_model(custom_expr, param_names):
    """
    Create a callable custom model function from a user-defined expression.
    Example: "X0 * exp(mu * t)" becomes a function that computes X0 * exp(mu * t).
    """
    if not custom_expr:
        st.error("Custom model expression is empty. Please provide a valid expression.")
        return None
    try:
        # Define symbols
        t = sp.symbols('t')
        params = sp.symbols(param_names)
        
        # Parse the expression
        expr = sp.sympify(custom_expr)
        
        # Create a lambda function that accepts t and individual parameters
        func = sp.lambdify([t] + list(params), expr, 'numpy')
        
        return func
    except Exception as e:
        st.error(f"Error parsing custom model expression: {e}")
        return None


# Create example data dynamically (if needed)
def create_example_data():
    example_data = {
        "Time": [0, 1, 2, 3, 4],
        "A1": [0.1, 0.15, 0.2, 0.25, 0.3],
        "A2": [0.2, 0.25, 0.3, 0.35, 0.4],
        "B1": [0.3, 0.35, 0.4, 0.45, 0.5],
    }
    df = pd.DataFrame(example_data)
    return df

default_guesses = {
    "Exponential Growth": [0.1, 1.0],
    "Logistic Growth": [0.1, 1.0, 2.0],
    "Baranyi Growth": [1.0, 0.1, 0.1],
    "Lag-Exponential-Saturation Growth": [0.1, 1.0, 0.1, 2.0]
}

# Update MODEL_PARAMS to include 'Custom Function'
MODEL_PARAMS = {
    "Polynomial Growth": ["a", "n", "b"],
    "Polynomial Function": ["a", "b", "c"],
    "Exponential Growth": ["mu", "X0"],
    "Logistic Growth": ["mu", "X0", "K"],
    "Baranyi Growth": ["X0", "mu", "q0"],
    "Lag-Exponential-Saturation Growth": ["mu", "X0", "q0", "K"],
    "Custom Function": [],  # Parameters will be dynamically determined
    "Automatic Fit": []  # Placeholder for automatic fit
}

# Update MODEL_FUNCTIONS to include a placeholder for 'Custom Function'
MODEL_FUNCTIONS = {
    "Exponential Growth": exponential_growth,
    "Logistic Growth": logistic_growth,
    "Baranyi Growth": baranyi_growth,
    "Lag-Exponential-Saturation Growth": lag_exponential_saturation_growth,
    "Custom Function": None,  # Will be handled separately
    "Automatic Fit": None  # Placeholder for automatic fit
}

# Define parameter units mapping
PARAMETER_UNITS = {
    "mu": "[1/time]",
    "X0": "[OD]",
    "K": "[OD]"
}

def load_json_config(json_file):
    """
    Safely parse an uploaded JSON file and return it as a Python dictionary.
    Returns None if parsing fails or if no file is provided.
    """
    if json_file is not None:
        import json
        try:
            config_data = json.load(json_file)
            return config_data
        except Exception as e:
            st.error(f"Error parsing JSON file: {e}")
    return None

def main():
    # Page configuration
    st.set_page_config(page_title="Bacterial Growth Analysis", page_icon="🔬", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bacterial Growth Analysis</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "phases" not in st.session_state:
        st.session_state["phases"] = []  # Initialize an empty list for phases
    if "ode_phases" not in st.session_state:
        st.session_state["ode_phases"] = []  # Initialize an empty list for ODE phases
    if "groups_data" not in st.session_state:
        st.session_state["groups_data"] = {}
    if "selected_sample_wells_by_group" not in st.session_state:
        st.session_state["selected_sample_wells_by_group"] = {}

    
    # Step 1: Layout Selection
    rows, columns = select_layout()
    labels = generate_labels(rows, columns)
    
    # Initialize `num_groups` and `groups_data` to avoid UnboundLocalError
    num_groups = 1
    groups_data = {}
    df = None  # Initialize df to avoid reference errors

    # Use tabs to organize the main sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Upload Data", "Background Subtraction", "Operations", "Fitting", "Custom ODE Analysis", "Automatic Phase Detection", "Growth Models"])

    # Step 2: File Upload and Data Processing
    # Tab 1: Upload Data
    # In Tab 1 (Upload Data)
    with tab1:
        st.header("📁 Upload and Inspect Data")

        # Define file paths
        example_file_path = os.path.join("assests", "example_spreadsheet.xlsx")
        default_layout_image_path = os.path.join("assests", "image.png")

        # Initial layout: Left (Example File) | Right (Image)
        col_left, col_right = st.columns(2)

        # Left Column: Example File Download
        with col_left:
            st.subheader("📄 Example Spreadsheet")
            if os.path.exists(example_file_path):
                with open(example_file_path, "rb") as file:
                    example_bytes = file.read()
                st.download_button(
                    label="Download Example Spreadsheet",
                    data=example_bytes,
                    file_name="example_spreadsheet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("❌ Example file not found. Please check the file path.")

            st.write("""
            Please ensure that your file follows this format:
            
            - The first column should be labeled `Time` and contain the time points.
            - Subsequent columns should represent well measurements, like `A1`, `A2`,`B1`, `B2`, etc.
            """)

            # Display the example DataFrame in the first column
            sample_file = create_example_data()
            st.dataframe(sample_file)

        # Right Column: Plate Reader Layout Image
        with col_right:
            st.subheader("Plate Reader Layout")

            if os.path.exists(default_layout_image_path):
                default_image = Image.open(default_layout_image_path)
                
                # Resize the image (Adjust width & height as needed)
                max_width = 500  # Adjust the max width (pixels)
                max_height = 500  # Adjust the max height (pixels)
                default_image.thumbnail((max_width, max_height))  # Resize while maintaining aspect ratio
                
                st.image(default_image, caption="Default Plate Reader Layout", use_container_width=False)
            else:
                st.info("⚠️ No default layout image available.")

        # --- File Upload Section ---
        st.subheader("📂 Upload Your Data File")
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV or XLSX)",
            type=["xlsx", "csv"],
            key="data_file"
        )

        df = None
        if uploaded_file is not None:
            with st.spinner("🔄 Reading and processing the data..."):
                df = read_data(uploaded_file, rows, columns)
                if df is not None:
                    st.session_state['df'] = df
                    with st.expander("📊 View Raw Data"):
                        st.write(df)
        else:
            st.info("📌 Please upload a data file (CSV or XLSX) to continue.")

        # --- Manual Well Selection (Only if Data is Loaded) ---
        if df is not None:
            st.subheader("🔬 Select Wells to Plot Manually")
            selected_wells = create_button_layout(rows, columns, labels, key_prefix="tab1_wells")

            if selected_wells:
                st.success(f"✅ Selected Wells: {', '.join(selected_wells)}")
                plot_selected_wells(df, selected_wells)
            else:
                st.info("🛑 Please select some wells.")

    # -------------------------------
    # Tab 2: Background Subtraction
    # -------------------------------
    # Tab 2: Background Subtraction
    with tab2:
        st.header("Background Subtraction")

        # Check that data is uploaded.
        if 'df' not in st.session_state or st.session_state['df'] is None:
            st.info("Please upload a data file in Tab 1 first.")
        else:
            df = st.session_state['df']

            # Initialize required session state keys.
            if "selected_sample_wells_by_group" not in st.session_state:
                st.session_state["selected_sample_wells_by_group"] = {}
            if "groups_data" not in st.session_state:
                st.session_state["groups_data"] = {}

            # Retrieve the plate layout (or default to 8x12) and generate labels.
            rows = st.session_state.get("rows", 8)
            columns = st.session_state.get("columns", 12)
            labels = generate_labels(rows, columns)

            # ----------------------------------------------------------------------
            # Section 1: JSON Upload for Configuration (Automatic Processing)
            # ----------------------------------------------------------------------
            st.subheader("🔄 Upload JSON Configuration (Optional)")
            uploaded_json_config = st.file_uploader("Choose JSON Configuration File", type=["json"], key="tab2_json_upload")
            json_loaded = False

            if uploaded_json_config is not None:
                try:
                    config_data = json.load(uploaded_json_config)
                    st.success("✅ JSON configuration loaded successfully.")
                    json_loaded = True

                    # Update plate layout if provided.
                    layout = config_data.get("plate_layout", {})
                    json_rows = layout.get("rows", rows)
                    json_columns = layout.get("columns", columns)
                    if json_rows != rows or json_columns != columns:
                        st.warning(f"JSON layout ({json_rows}x{json_columns}) differs from current layout ({rows}x{columns}). Using JSON layout.")
                        rows, columns = json_rows, json_columns
                        st.session_state["rows"] = rows
                        st.session_state["columns"] = columns
                        labels = generate_labels(rows, columns)

                    # Update number of groups.
                    num_groups = config_data.get("num_groups", 1)
                    st.session_state["num_groups"] = num_groups
                    st.info(f"📊 Number of Groups (from JSON): {num_groups}")

                    groups_data = {}

                    # Process each group.
                    for group_num in range(1, num_groups + 1):
                        group_str = str(group_num)
                        group_info = config_data.get("groups", {}).get(group_str, {})
                        blank_wells = group_info.get("blank_wells", [])
                        sample_wells = group_info.get("sample_wells", [])
                        fitting_info = group_info.get("fitting", {})  # Fitting info (e.g., model, initial guesses)

                        # Update session state with JSON values.
                        st.session_state[f"group_{group_num}_blank_wells"] = blank_wells
                        st.session_state[f"group_{group_num}_sample_wells"] = sample_wells
                        st.session_state["selected_sample_wells_by_group"][group_num] = sample_wells

                        st.write(f"**Group {group_num} (from JSON):**")
                        st.write(f"Blank Wells: {', '.join(blank_wells) if blank_wells else 'None'}")
                        st.write(f"Sample Wells: {', '.join(sample_wells) if sample_wells else 'None'}")
                        st.write(f"Fitting Info: {fitting_info if fitting_info else 'Not specified'}")

                        if blank_wells:
                            st.write("🔍 Preview of a Blank Well:")
                            display_single_well_preview(df, blank_wells[0])
                        if sample_wells:
                            st.write("🔍 Preview of a Sample Well:")
                            display_single_well_preview(df, sample_wells[0])
                        
                        # Perform background subtraction.
                        groups_data = perform_background_subtraction(groups_data, df, group_num, blank_wells, sample_wells)
                        
                        # --- Fitting Process for Blank Wells ---
                        if blank_wells:
                            avg_blank = df[blank_wells].mean(axis=1)
                            # Use model info from JSON, if provided; defaults otherwise.
                            model_used = fitting_info.get("model", "Polynomial Function")
                            initial_guesses = fitting_info.get("initial_guesses", [1.0, 1.0, 1.0])
                            if model_used == "Polynomial Growth":
                                model_func = polynomial_growth
                            else:
                                model_func = polynomial_func
                            try:
                                popt, pcov = curve_fit(model_func, df['Time'], avg_blank, p0=initial_guesses)
                                y_fit = model_func(df['Time'], *popt)
                                st.success(f"Fitting for Group {group_num} blank wells successful.")
                                
                                # Compute confidence intervals.
                                dof = len(df['Time']) - len(popt)
                                residual_variance = np.var(avg_blank - y_fit, ddof=len(popt))
                                lower_bound, upper_bound = compute_confidence_intervals(df['Time'], popt, pcov, 0.05, dof, residual_variance, model_func)
                                
                                # Plot the blank fit with confidence intervals.
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df['Time'], y=avg_blank, mode='lines', name='Observed Blank'))
                                fig.add_trace(go.Scatter(x=df['Time'], y=y_fit, mode='lines', name='Fitted Curve', line=dict(color='red')))
                                fig.add_trace(go.Scatter(x=df['Time'], y=upper_bound, mode='lines', name='Upper CI', line=dict(color='rgba(255,0,0,0.2)')))
                                fig.add_trace(go.Scatter(x=df['Time'], y=lower_bound, mode='lines', name='Lower CI', line=dict(color='rgba(255,0,0,0.2)')))
                                fig.update_layout(title=f"Fitting for Group {group_num} Blank Wells",
                                                xaxis_title='Time', yaxis_title='OD', template='plotly_white')
                                st.plotly_chart(fig)
                            except Exception as e_fit:
                                st.error(f"Fitting failed for Group {group_num} blank wells: {e_fit}")
                        
                        # --- Now, after blank well fitting, process sample well background subtraction and plots ---
                        if sample_wells:
                            group_df = groups_data.get(f"Group_{group_num}_bg_subtracted", None)
                            if group_df is not None:
                                st.write(f"### Background-Corrected Data for Group {group_num}")
                                st.write(group_df)
                                plot_selected_wells(group_df, sample_wells)
                                plot_avg_sd_bg_subtracted(group_df, sample_wells, group_num)
                            else:
                                st.warning(f"No background subtraction data for Group {group_num}.")
                    
                    st.session_state["groups_data"] = groups_data

                except Exception as e:
                    st.error(f"❌ Failed to load or parse the JSON file. Error: {e}")

            # ----------------------------------------------------------------------
            # Section 2: Manual Configuration (if no JSON file is uploaded)
            # ----------------------------------------------------------------------
            if not json_loaded:
                st.subheader("🔢 Specify Number of Groups for Background Correction")
                num_groups = st.number_input("Number of Groups:", min_value=1, step=1, value=1, key="tab2_num_groups")
                st.session_state["num_groups"] = num_groups
                groups_data = st.session_state["groups_data"]

                for group_num in range(1, num_groups + 1):
                    st.subheader(f"Group {group_num} Well Selection for Background Correction")
                    with st.expander(f"Group {group_num} Blank and Sample Well Selection", expanded=True):
                        st.write(f"🔘 Select blank wells for Group {group_num}")
                        blank_wells_key = f"group_{group_num}_blank_wells"
                        selected_blank_wells = create_button_layout(rows, columns, labels, key_prefix=blank_wells_key)
                        st.session_state[f"group_{group_num}_blank_wells"] = selected_blank_wells
                        st.write("DEBUG - Blank Wells:", st.session_state.get(f"group_{group_num}_blank_wells", []))
                        
                        blank_preview = st.empty()
                        if selected_blank_wells:
                            blank_preview.markdown(f"Preview of {selected_blank_wells[-1]}")
                            display_single_well_preview(df, selected_blank_wells[-1])
                        else:
                            blank_preview.empty()
                        
                        if selected_blank_wells:
                            st.subheader(f"Fit Model to Blank Wells - Group {group_num}")
                            plot_average(df, selected_blank_wells)
                            selected_model = st.selectbox(
                                f"Select Model for Blank Well Fitting - Group {group_num}",
                                ["Polynomial Growth", "Polynomial Function"],
                                key=f"model_{group_num}"
                            )
                            # Save the fitting model info in session state (optional)
                            st.session_state[f"group_{group_num}_fitting"] = {"model": selected_model, "initial_guesses": [1.0, 1.0, 1.0]}
                            
                            avg_blank = df[selected_blank_wells].mean(axis=1)
                            if selected_model == "Polynomial Growth":
                                model_func = polynomial_growth
                            else:
                                model_func = polynomial_func
                            try:
                                popt, pcov = curve_fit(model_func, df['Time'], avg_blank)
                                y_pred = model_func(df['Time'], *popt)
                                st.success(f"Fitting for Group {group_num} blank wells successful.")
                                
                                # Create and show a DataFrame of fitted parameters.
                                #MODEL_PARAMS = {"Polynomial Growth": ["a", "b", "c"], "Polynomial Function": ["a", "b", "c"]}
                                param_names = MODEL_PARAMS[selected_model]
                                param_df = pd.DataFrame({
                                    "Parameter": param_names,
                                    "Value": popt
                                })
                                st.write(f"### Fitted Parameters for Group {group_num} ({selected_model})")
                                st.dataframe(param_df)
                                
                                # Compute Confidence Intervals for the fit.
                                dof = len(df['Time']) - len(popt)
                                residual_variance = np.var(avg_blank - y_pred, ddof=len(popt))
                                lower_bound, upper_bound = compute_confidence_intervals(df['Time'], popt, pcov, 0.05, dof, residual_variance, model_func)
                                
                                # Plot CI with fitted curve.
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df['Time'], y=avg_blank, mode='lines', name='Observed Blank'))
                                fig.add_trace(go.Scatter(x=df['Time'], y=y_pred, mode='lines', name='Fitted Curve', line=dict(color='red')))
                                fig.add_trace(go.Scatter(x=df['Time'], y=upper_bound, mode='lines', name='Upper CI', line=dict(color='rgba(255,0,0,0.2)')))
                                fig.add_trace(go.Scatter(x=df['Time'], y=lower_bound, mode='lines', name='Lower CI', line=dict(color='rgba(255,0,0,0.2)')))
                                fig.update_layout(title=f"Fitting for Group {group_num} Blank Wells",
                                                xaxis_title='Time', yaxis_title='OD', template='plotly_white')
                                st.plotly_chart(fig)
                            except Exception as e_fit:
                                st.error(f"Fitting failed for Group {group_num} blank wells: {e_fit}")

                        
                        st.write(f"🧪 Select sample wells for Group {group_num}")
                        sample_wells_key = f"group_{group_num}_sample_wells"
                        selected_sample_wells = create_button_layout(rows, columns, labels, key_prefix=sample_wells_key)
                        st.session_state[f"group_{group_num}_sample_wells"] = selected_sample_wells
                        st.session_state["selected_sample_wells_by_group"][group_num] = selected_sample_wells
                        st.write("DEBUG - Sample Wells:", st.session_state.get(f"group_{group_num}_sample_wells", []))
                        
                        sample_preview = st.empty()
                        if selected_sample_wells:
                            sample_preview.markdown(f"Preview of {selected_sample_wells[-1]}")
                            display_single_well_preview(df, selected_sample_wells[-1])
                        else:
                            sample_preview.empty()
                        
                        if selected_sample_wells:
                            groups_data = perform_background_subtraction(groups_data, df, group_num, selected_blank_wells, selected_sample_wells)
                            st.session_state["groups_data"] = groups_data
                            group_df = groups_data.get(f"Group_{group_num}_bg_subtracted", None)
                            if group_df is not None:
                                st.write(f"### Background-Corrected Data for Group {group_num}")
                                st.write(group_df)
                                plot_selected_wells(group_df, selected_sample_wells)
                                plot_avg_sd_bg_subtracted(group_df, selected_sample_wells, group_num)
                            else:
                                st.warning("No data available after background subtraction.")
                        else:
                            st.warning(f"Group {group_num} sample wells have not been selected. Cannot perform background subtraction.")

            # ----------------------------------------------------------------------
            # Section 3: Build and Display Editable JSON Configuration for Download
            # ----------------------------------------------------------------------
            st.subheader("📥 Download Background Subtraction Configuration")
            if 'groups_data' in st.session_state and st.session_state['groups_data']:
                config_bg_subtraction = {
                    "plate_layout": {
                        "rows": rows,
                        "columns": columns
                    },
                    "num_groups": st.session_state.get("num_groups", 1),
                    "groups": {}
                }
                for group_num in range(1, st.session_state.get("num_groups", 1) + 1):
                    group_key = str(group_num)
                    blank_wells = st.session_state.get(f"group_{group_num}_blank_wells", [])
                    sample_wells = st.session_state.get(f"group_{group_num}_sample_wells", [])
                    # Include the fitting info if available.
                    fitting_info = st.session_state.get(f"group_{group_num}_fitting", {})
                    config_bg_subtraction["groups"][group_key] = {
                        "blank_wells": blank_wells,
                        "sample_wells": sample_wells,
                        "fitting": fitting_info
                    }
                editable_json = json.dumps(config_bg_subtraction, indent=2)
                edited_bg_json = st.text_area("Edit Background Subtraction JSON Configuration:", value=editable_json, height=200)
                if st.button("Download Background Subtraction JSON"):
                    st.download_button(
                        label="💾 Download JSON",
                        data=edited_bg_json,
                        file_name="background_subtraction_config.json",
                        mime="application/json"
                    )
            else:
                st.info("No background subtraction data available to download.")

    # Step 7: Operations on Background-Subtracted Groups (Handles Single and Multiple Groups)
    # Operations Tab
    with tab3:
        st.subheader("Operations on Background-Subtracted Data")

        if num_groups == 1 and groups_data:
            # Single group scenario
            st.info("Only one group is selected. Operations will be performed on the background-subtracted data of this group.")

            group1_data = groups_data.get("Group_1_bg_subtracted")
            sample_wells_group1 = st.session_state.get("selected_sample_wells_by_group", {}).get(1, [])
            if group1_data is not None and sample_wells_group1:
                st.write("Background-subtracted data for Group 1 (treated as operated data):")
                st.write(group1_data)

                # Plot data for Group 1
                st.subheader("Plot Background-Subtracted Data for Group 1")
                plot_selected_wells(group1_data, sample_wells_group1)
                plot_avg_sd_bg_subtracted(group1_data, sample_wells_group1, group_num=1)

                # Set operated data for phase analysis
                operated_data = group1_data
                # Here, group1_data might already have 'Average' column added earlier

                # Update selected_operated_wells
                selected_operated_wells = sample_wells_group1
                st.session_state["operated_data"] = operated_data
                st.session_state["selected_operated_wells"] = selected_operated_wells

        elif num_groups > 1 and groups_data:
            # Multiple groups scenario
            group1_data = groups_data.get("Group_1_bg_subtracted")
            group2_data = groups_data.get("Group_2_bg_subtracted")

            if group1_data is not None and group2_data is not None:
                operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])
                sample_wells_group1 = st.session_state.get("selected_sample_wells_by_group", {}).get(1, [])
                sample_wells_group2 = selected_sample_wells_by_group.get(2, [])
                
                if sample_wells_group1 and sample_wells_group2:
                    group1_data_samples = group1_data[["Time"] + sample_wells_group1]
                    group2_data_samples = group2_data[["Time"] + sample_wells_group2]

                    operated_data = perform_group_operations(group1_data_samples, group2_data_samples, operation)
                    st.write(f"Result of {operation} operation between Group 1 and Group 2")
                    st.write(operated_data)

                    # IMPORTANT: Add 'Average' column before plotting
                    # After performing the operation, 'Average' is not yet defined
                    # We need to compute it now:
                    if operated_data.shape[1] > 1:
                        operated_data["Average"] = operated_data.iloc[:, 1:].mean(axis=1)

                    # Plot operated data
                    st.subheader("Plot Operated Data")
                    plot_selected_wells(operated_data, operated_data.columns[1:])
                    plot_avg_sd_operated(operated_data, operated_data.columns[1:])

                    # Update selected_operated_wells
                    selected_operated_wells = operated_data.columns[1:].tolist()
                    st.session_state["operated_data"] = operated_data
                    st.session_state["selected_operated_wells"] = selected_operated_wells

        else:
            st.warning("No background-subtracted data is available for operations. Please ensure background correction is completed in the 'Background Correction' tab.")





    # Phase Analysis Tab
    with tab4:
        st.subheader("Fitting")

        # --- Use Operated Data from Tab 2 if Tab 3 Was Skipped ---
        if "operated_data" not in st.session_state or st.session_state["operated_data"] is None:
            groups_data = st.session_state.get("groups_data", {})
            if st.session_state.get("num_groups", 1) == 1 and "Group_1_bg_subtracted" in groups_data:
                st.session_state["operated_data"] = groups_data.get("Group_1_bg_subtracted")
            else:
                st.error("Operated data is not available. Please complete Tab 2 (and Tab 3 if using multiple groups).")
        operated_data = st.session_state.get("operated_data")
        selected_operated_wells = st.session_state.get("selected_operated_wells", [])

        if operated_data is None or not selected_operated_wells:
            st.warning("Operated data or selected wells are not available. Please verify your inputs in Tab 2 (and Tab 3 if applicable).")
        else:
            if "Average" not in operated_data.columns:
                operated_data["Average"] = operated_data[selected_operated_wells].mean(axis=1)
            st.info("Operated Data Preview:")
            st.dataframe(operated_data.head())

            # ---- Overall Operated Data Plot ----
            st.subheader("Average and Standard Deviation of Operated Data (All Selected Wells)")
            try:
                plot_avg_sd_operated(operated_data, selected_operated_wells)
            except Exception as e:
                st.error(f"Error plotting operated data: {e}")

            ##############################
            # AUTOMATIC JSON FITTING
            ##############################
            st.markdown("### Automatic Fitting via JSON Configuration")
            uploaded_json_fit = st.file_uploader("Upload JSON for Automatic Fitting", type=["json"], key="tab4_json_upload")
            if uploaded_json_fit is not None:
                try:
                    fit_config = json.load(uploaded_json_fit)
                    auto_fits = fit_config.get("fit_configuration", [])
                    if not auto_fits:
                        st.warning("No fit configurations found in the JSON file.")
                    else:
                        for idx, fit_conf in enumerate(auto_fits):
                            st.markdown(f"#### Processing Fit {idx+1}")
                            time_interval = fit_conf.get("time_interval", {})
                            start = time_interval.get("start")
                            end = time_interval.get("end")
                            model = fit_conf.get("model")
                            initial_guesses = fit_conf.get("initial_guesses")
                            bounds = fit_conf.get("bounds")  # e.g., {"lower": [...], "upper": [...]}
                            custom_model_expr = fit_conf.get("custom_model_expr")
                            custom_params = fit_conf.get("custom_params")
                            wells = fit_conf.get("selected_operated_wells", selected_operated_wells)

                            if start is None or end is None or model is None or initial_guesses is None:
                                st.error("Missing one or more required fields in fit configuration (time_interval, model, initial_guesses).")
                                continue

                            phase_data = operated_data[(operated_data["Time"] >= start) & (operated_data["Time"] <= end)][["Time"] + wells]
                            phase_data = phase_data.dropna(subset=wells)
                            if phase_data.empty:
                                st.warning(f"No data points found in the interval {start} to {end}.")
                                continue
                            phase_data["Average"] = phase_data[wells].mean(axis=1)
                            time_vals = phase_data["Time"].values
                            y_data = phase_data["Average"].values

                            if model == "Custom Function":
                                model_func = create_custom_model(custom_model_expr, custom_params)
                                param_names = [p.strip() for p in custom_params if p.strip()]
                            else:
                                model_func = MODEL_FUNCTIONS.get(model)
                                param_names = MODEL_PARAMS.get(model, [])
                            if model_func is None:
                                st.error(f"No model function available for model {model}.")
                                continue

                            try:
                                if bounds:
                                    lower_bounds = bounds.get("lower")
                                    upper_bounds = bounds.get("upper")
                                    popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
                                else:
                                    popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses)
                            except Exception as e_fit:
                                st.error(f"Fitting failed for interval {start}-{end}: {e_fit}")
                                continue

                            y_pred = model_func(time_vals, *popt)
                            residuals = y_data - y_pred
                            residual_variance = np.var(residuals, ddof=len(popt))
                            dof = len(y_data) - len(popt)
                            lower_bound_ci, upper_bound_ci = compute_confidence_intervals(time_vals, popt, pcov, 0.05, dof, residual_variance, model_func)
                            perr = np.sqrt(np.diag(pcov))

                            RSS = np.sum(residuals**2)
                            AIC = 2 * len(popt) + len(y_data) * np.log(RSS/len(y_data))
                            t_statistic = popt / perr
                            p_values = 2 * (1 - t_dist.cdf(np.abs(t_statistic), df=dof))

                            phase_dict = {
                                "id": str(uuid.uuid4()),
                                "phase": idx + 1,
                                "time_interval": {"start": start, "end": end},
                                "selected_operated_wells": wells,
                                "model": model,
                                "parameters": param_names,
                                "initial_guesses": initial_guesses,
                                "bounds": bounds,
                            }
                            fit_results = {
                                "phase_time": time_vals,
                                "fit": y_pred,
                                "lower_bound": lower_bound_ci,
                                "upper_bound": upper_bound_ci,
                                "std_dev": phase_data[wells].std(axis=1).values,
                                "parameters": popt,
                                "param_errors": perr,
                                "AIC": AIC,
                                "t_statistic": t_statistic,
                                "p_values": p_values
                            }
                            phase_dict["fit_results"] = fit_results
                            phase_dict["phase_time"] = time_vals
                            phase_dict["fit"] = y_pred
                            phase_dict["lower_bound"] = lower_bound_ci
                            phase_dict["upper_bound"] = upper_bound_ci
                            phase_dict["std_dev"] = phase_data[wells].std(axis=1).values
                            phase_dict["parameters"] = popt
                            phase_dict["param_errors"] = perr
                            phase_dict["AIC"] = AIC
                            phase_dict["t_statistic"] = t_statistic
                            phase_dict["p_values"] = p_values

                            st.session_state.setdefault("phases", []).append(phase_dict)
                            st.success(f"Fit {idx+1} completed for interval {start} to {end} using model {model}.")
                            plot_fitted_curves(phase_data, time_vals, y_data, y_pred, model)
                            plot_confidence_intervals(phase_data, lower_bound_ci, upper_bound_ci, y_pred, phase_data[wells].std(axis=1))
                            parameter_labels = [f"{p} {PARAMETER_UNITS.get(p, '')}" for p in param_names]
                            param_table = pd.DataFrame({
                                "Parameter": parameter_labels,
                                "Estimate": popt,
                                "Std. Error": perr,
                                "t-Statistic": t_statistic,
                                "p-Value": p_values
                            })
                            st.dataframe(param_table)
                except Exception as e:
                    st.error(f"Error processing JSON file: {e}")
            else:
                st.info("No JSON configuration uploaded for automatic fitting. You can add fits manually below.")

            ##############################
            # MANUAL FITTING UI
            ##############################
            st.markdown("### Add a New Fit Manually")
            col1, col2 = st.columns(2)
            with col1:
                new_fit_start = st.number_input(
                    "Start Time for New Fit",
                    min_value=float(operated_data["Time"].min()),
                    max_value=float(operated_data["Time"].max()),
                    step=0.1,
                    key="new_fit_start",
                    format="%.5f"
                )
            with col2:
                new_fit_end = st.number_input(
                    "End Time for New Fit",
                    min_value=new_fit_start,
                    max_value=float(operated_data["Time"].max()),
                    step=0.1,
                    key="new_fit_end",
                    format="%.5f"
                )
            if st.button("Add New Fit", key="add_new_fit"):
                new_phase = {
                    "id": str(uuid.uuid4()),
                    "phase": len(st.session_state.get("phases", [])) + 1,
                    "time_interval": {"start": new_fit_start, "end": new_fit_end},
                    "selected_operated_wells": selected_operated_wells,
                    "model": None,
                    "parameters": [],
                    "initial_guesses": [],
                    "bounds": None,
                    "fit_results": None
                }
                st.session_state.setdefault("phases", []).append(new_phase)
                st.success(f"Added new fit entry for interval {new_fit_start} to {new_fit_end}.")

            for i, phase in enumerate(st.session_state.get("phases", [])):
                # Ensure each phase has a stable id
                if "id" not in phase:
                    phase["id"] = str(uuid.uuid4())
                phase_id = phase["id"]
                with st.expander(f"Manual Fit {i+1}"):
                    try:
                        t_interval = phase.get("time_interval", {})
                        fit_start = st.text_input(
                            f"Start Time for Fit {i+1}",
                            value=str(t_interval.get("start", operated_data['Time'].min())),
                            key=f"start_{i}"
                        )
                        fit_end = st.text_input(
                            f"End Time for Fit {i+1}",
                            value=str(t_interval.get("end", operated_data['Time'].max())),
                            key=f"end_{i}"
                        )
                        phase["time_interval"] = {"start": float(fit_start), "end": float(fit_end)}

                        # Delete Fit option using stable id
                        if st.button(f"Delete Fit {i+1}", key=f"delete_phase_{phase_id}"):
                            st.session_state["phases"].pop(i)
                            st.success(f"Deleted Fit {i+1}")
                            return  # Use return() here to halt further processing

                        phase_data = operated_data[(operated_data["Time"] >= float(fit_start)) &
                                                (operated_data["Time"] <= float(fit_end))]
                        if phase_data.empty:
                            st.warning("No data points found in this interval.")
                            continue
                        phase_data["Average"] = phase_data[selected_operated_wells].mean(axis=1)
                        time_vals = phase_data["Time"].values
                        y_data = phase_data["Average"].values

                        phase["model"] = st.selectbox(
                            f"Select Model for Fit {i+1}",
                            ["Exponential Growth", "Logistic Growth", "Baranyi Growth", "Lag-Exponential-Saturation Growth", "Custom Function", "Automatic Fit"],
                            key=f"model_{i}"
                        )
                        if phase["model"] == "Custom Function":
                            phase["custom_model_expr"] = st.text_input(
                                f"Custom Model Expression for Fit {i+1}",
                                value=phase.get("custom_model_expr", "X * exp(mu * t)"),
                                key=f"custom_model_expr_{i}"
                            )
                            phase["custom_params"] = st.text_input(
                                f"Parameters to Optimize (comma-separated) for Fit {i+1}",
                                value=", ".join(phase.get("custom_params", [])) or "X, mu",
                                key=f"custom_params_{i}"
                            ).split(",")
                            model_func = create_custom_model(phase["custom_model_expr"],
                                                            [p.strip() for p in phase["custom_params"] if p.strip()])
                            phase["parameters"] = [p.strip() for p in phase["custom_params"] if p.strip()]
                        elif phase["model"] == "Automatic Fit":
                            candidate_models = [m for m in MODEL_FUNCTIONS.keys() if m not in ["Custom Function", "Automatic Fit"]]
                            st.info("Automatic Fit: Evaluating candidate models.")
                            best_model = None
                            best_popt = None
                            best_pcov = None
                            best_aic = np.inf
                            best_candidate = None
                            for candidate in candidate_models:
                                try:
                                    cf_model_func = MODEL_FUNCTIONS[candidate]
                                    guesses = default_guesses.get(candidate, [1.0] * len(MODEL_PARAMS.get(candidate, [])))
                                    popt_candidate, pcov_candidate = curve_fit(cf_model_func, time_vals, y_data, p0=guesses)
                                    y_pred_candidate = cf_model_func(time_vals, *popt_candidate)
                                    residuals_candidate = y_data - y_pred_candidate
                                    rss_candidate = np.sum(residuals_candidate**2)
                                    dof_candidate = len(y_data) - len(popt_candidate)
                                    aic_candidate = 2 * len(popt_candidate) + len(y_data) * np.log(rss_candidate/len(y_data))
                                    st.write(f"Candidate {candidate}: AIC = {aic_candidate:.2f}")
                                    if aic_candidate < best_aic:
                                        best_aic = aic_candidate
                                        best_model = cf_model_func
                                        best_popt = popt_candidate
                                        best_pcov = pcov_candidate
                                        best_candidate = candidate
                                except Exception as e_cand:
                                    st.warning(f"Candidate {candidate} failed: {e_cand}")
                                    continue
                            if best_model is None:
                                st.error("Automatic Fit could not find a suitable model.")
                                continue
                            st.success(f"Automatic Fit selected: {best_candidate} (AIC = {best_aic:.2f})")
                            model_func = best_model
                            phase["model"] = best_candidate
                            phase["parameters"] = MODEL_PARAMS.get(best_candidate, [])
                            popt = best_popt
                            pcov = best_pcov
                            y_pred = model_func(time_vals, *popt)
                        else:
                            model_func = MODEL_FUNCTIONS.get(phase["model"])
                            phase["parameters"] = MODEL_PARAMS.get(phase["model"], [])
                        if model_func is None:
                            st.error("No model function available.")
                            continue

                        st.markdown("#### Enter Initial Guesses and Optional Bounds")
                        params_list = phase.get("parameters", [])
                        initial_guesses_list = []
                        lower_bounds = []
                        upper_bounds = []
                        use_bounds = False
                        for param in params_list:
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                guess = st.number_input(f"Initial guess for {param}", value=1.0, key=f"{param}_{i}_guess", format="%.5f")
                                initial_guesses_list.append(guess)
                            with col_b:
                                bound_flag = st.checkbox(f"Use bounds for {param}?", value=False, key=f"use_bounds_{param}_{i}")
                            if bound_flag:
                                use_bounds = True
                                with col_c:
                                    lb = st.number_input(f"Lower bound for {param}", value=0.0, key=f"lb_{param}_{i}", format="%.5f")
                                    lower_bounds.append(lb)
                                with col_d:
                                    ub = st.number_input(f"Upper bound for {param}", value=10.0, key=f"ub_{param}_{i}", format="%.5f")
                                    upper_bounds.append(ub)
                            else:
                                lower_bounds.append(-np.inf)
                                upper_bounds.append(np.inf)
                        phase["initial_guesses"] = initial_guesses_list
                        phase["bounds"] = {"lower": lower_bounds, "upper": upper_bounds} if use_bounds else None

                        if st.button(f"Fit Model for Fit {i+1}", key=f"fit_model_{i}"):
                            try:
                                if phase["bounds"]:
                                    popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses_list, bounds=(lower_bounds, upper_bounds))
                                else:
                                    popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses_list)
                                y_pred = model_func(time_vals, *popt)
                                residuals = y_data - y_pred
                                residual_variance = np.var(residuals, ddof=len(popt))
                                dof = len(y_data) - len(popt)
                                lower_bound_ci, upper_bound_ci = compute_confidence_intervals(time_vals, popt, pcov, 0.05, dof, residual_variance, model_func)
                                perr = np.sqrt(np.diag(pcov))
                                
                                RSS = np.sum(residuals**2)
                                AIC = 2 * len(popt) + len(y_data) * np.log(RSS/len(y_data))
                                t_statistic = popt / perr
                                p_values = 2 * (1 - t_dist.cdf(np.abs(t_statistic), df=dof))
                                
                                phase["fit_results"] = {
                                    "phase_time": time_vals,
                                    "fit": y_pred,
                                    "lower_bound": lower_bound_ci,
                                    "upper_bound": upper_bound_ci,
                                    "std_dev": phase_data[selected_operated_wells].std(axis=1).values,
                                    "parameters": popt,
                                    "param_errors": perr,
                                    "AIC": AIC,
                                    "t_statistic": t_statistic,
                                    "p_values": p_values
                                }
                                phase.setdefault("phase", i+1)
                                phase["phase_time"] = time_vals
                                phase["fit"] = y_pred
                                phase["lower_bound"] = lower_bound_ci
                                phase["upper_bound"] = upper_bound_ci
                                phase["std_dev"] = phase_data[selected_operated_wells].std(axis=1).values
                                phase["parameters"] = popt
                                phase["param_errors"] = perr
                                phase["AIC"] = AIC
                                phase["t_statistic"] = t_statistic
                                phase["p_values"] = p_values

                                st.success("Model fitted successfully!")
                                plot_fitted_curves(phase_data, time_vals, y_data, y_pred, phase["model"])
                                plot_confidence_intervals(phase_data, lower_bound_ci, upper_bound_ci, y_pred, phase_data[selected_operated_wells].std(axis=1))
                                parameter_labels = [f"{p} {PARAMETER_UNITS.get(p, '')}" for p in phase.get("parameters", [])]
                                param_table = pd.DataFrame({
                                    "Parameter": parameter_labels,
                                    "Estimate": popt,
                                    "Std. Error": perr,
                                    "t-Statistic": t_statistic,
                                    "p-Value": p_values
                                })
                                st.dataframe(param_table)
                            except Exception as e_fit:
                                st.error(f"Error fitting model for Fit {i+1}: {e_fit}")
                    except Exception as e_manual:
                        st.error(f"Error in manual fitting for Fit {i+1}: {e_manual}")

            ##############################
            # SUMMARY PLOT AND JSON EXPORT
            ##############################
            st.markdown("### Summary of All Fit Results")
            if st.session_state.get("phases"):
                fitted_phases = [phase for phase in st.session_state["phases"] if phase.get("fit_results") is not None and "phase_time" in phase]
                if fitted_phases:
                    fig_summary = plot_phase_fit_with_ci(fitted_phases, operated_data, selected_operated_wells)
                    st.plotly_chart(fig_summary, use_container_width=True)
                else:
                    st.info("No fits available to generate a summary plot.")

                config_fits = {"fit_configuration": []}
                for phase in st.session_state["phases"]:
                    config_fits["fit_configuration"].append({
                        "time_interval": phase.get("time_interval"),
                        "selected_operated_wells": phase.get("selected_operated_wells"),
                        "model": phase.get("model"),
                        "initial_guesses": phase.get("initial_guesses"),
                        "bounds": phase.get("bounds")
                    })
                editable_json = json.dumps(config_fits, indent=2)
                edited_json = st.text_area("Edit Fit Configuration JSON:", value=editable_json, height=200)
                if st.button("Download Fit Configuration JSON"):
                    st.download_button(
                        label="💾 Download JSON",
                        data=edited_json,
                        file_name="fit_configuration.json",
                        mime="application/json"
                    )
            else:
                st.info("No fit results to summarize.")

    ###############################################################################
    # Tab 5: Custom ODE Fit Analysis (with full old-results display and safe key checks)
    ###############################################################################
    with tab5:
        st.header("📈 Custom ODE Fit Analysis (CI Only for X)")

        # 0) Initialize session state list for fits (if not done already)
        if "ode_fits" not in st.session_state:
            st.session_state["ode_fits"] = []

        # Retrieve operated data & selected wells
        operated_data = st.session_state.get("operated_data")
        selected_operated_wells = st.session_state.get("selected_operated_wells", [])

        if operated_data is None or not selected_operated_wells:
            st.warning("⚠️ Please complete previous steps to get operated data first.")
        else:
            # Ensure "Average" exists
            if "Average" not in operated_data.columns:
                operated_data["Average"] = operated_data[selected_operated_wells].mean(axis=1)

            # Quick overall plot
            st.subheader("Overall Operated Data")
            plot_avg_sd_operated(operated_data, selected_operated_wells)

            #----------------------------------------------------------------------
            # 1) Let user specify a time interval BEFORE adding new ODE fit
            #----------------------------------------------------------------------
            st.subheader("Specify Time Interval for a New ODE Fit")
            time_min = float(operated_data["Time"].min())
            time_max = float(operated_data["Time"].max())

            user_start = st.number_input(
                "New Fit Start Time",
                min_value=time_min,
                max_value=time_max,
                value=time_min,
                step=0.1,
                format="%.2f"
            )
            user_end = st.number_input(
                "New Fit End Time",
                min_value=user_start,
                max_value=time_max,
                value=time_max,
                step=0.1,
                format="%.2f"
            )

            if st.button("Add ODE Fit"):
                new_fit = {
                    "id": str(uuid.uuid4()),
                    "fit_start": user_start,
                    "fit_end": user_end,
                    # Default placeholders
                    "variables": ["X", "Y"],
                    "parameters": ["r", "k"],
                    "initial_conditions": [0.1, 0.5],
                    "odes": ["r * X", "-k * Y"],
                    "param_guesses": [0.2, 0.3],
                    "fit_results": None
                }
                st.session_state["ode_fits"].append(new_fit)
                st.success(f"Added new ODE fit for interval {user_start}–{user_end}.")

            #----------------------------------------------------------------------
            # 2) Loop over each ODE fit in session state
            #----------------------------------------------------------------------
            for i, fit in enumerate(st.session_state["ode_fits"]):
                fit_id = fit["id"]
                with st.expander(f"Fit {i+1}", expanded=True):
                    # A) Time interval
                    st.write("#### Time Interval")
                    c1, c2 = st.columns(2)
                    with c1:
                        new_start = st.number_input(
                            f"Start (Fit {i+1})",
                            min_value=time_min,
                            max_value=time_max,
                            value=float(fit["fit_start"]),
                            step=0.1,
                            format="%.2f",
                            key=f"start_{fit_id}"
                        )
                    with c2:
                        new_end = st.number_input(
                            f"End (Fit {i+1})",
                            min_value=new_start,
                            max_value=time_max,
                            value=float(fit["fit_end"]),
                            step=0.1,
                            format="%.2f",
                            key=f"end_{fit_id}"
                        )
                    fit["fit_start"] = new_start
                    fit["fit_end"] = new_end

                    # B) ODE definitions
                    st.write("#### Variables & ODEs")
                    var_str = st.text_input(
                        f"Variables (comma-separated) [Fit {i+1}]",
                        value=", ".join(fit["variables"]),
                        key=f"vars_{fit_id}"
                    )
                    fit["variables"] = [v.strip() for v in var_str.split(",") if v.strip()]

                    updated_odes = []
                    st.write("**ODE Expressions**")
                    for idx, var_name in enumerate(fit["variables"]):
                        default_expr = fit["odes"][idx] if idx < len(fit["odes"]) else ""
                        ode_expr = st.text_input(
                            f"d{var_name}/dt = ",
                            value=default_expr,
                            key=f"ode_expr_{fit_id}_{var_name}"
                        )
                        updated_odes.append(ode_expr)
                    fit["odes"] = updated_odes

                    if st.button(f"Show ODE Equations (Fit {i+1})", key=f"show_odes_{fit_id}"):
                        display_ode_equations(fit["variables"], fit["odes"])

                    param_str = st.text_input(
                        f"Parameters (comma-separated) [Fit {i+1}]",
                        value=", ".join(fit["parameters"]),
                        key=f"params_{fit_id}"
                    )
                    fit["parameters"] = [p.strip() for p in param_str.split(",") if p.strip()]

                    st.write("**Initial Conditions**")
                    new_inits = []
                    for idx, var_name in enumerate(fit["variables"]):
                        def_init = fit["initial_conditions"][idx] if idx < len(fit["initial_conditions"]) else 1.0
                        init_val = st.number_input(
                            f"Initial {var_name}",
                            value=def_init,
                            step=0.01,
                            format="%.5f",
                            key=f"init_{fit_id}_{var_name}"
                        )
                        new_inits.append(init_val)
                    fit["initial_conditions"] = new_inits

                    st.write("**Parameter Guesses**")
                    new_guesses = []
                    for idx, pname in enumerate(fit["parameters"]):
                        guess_def = fit["param_guesses"][idx] if idx < len(fit["param_guesses"]) else 1.0
                        guess_val = st.number_input(
                            f"Initial guess for {pname}",
                            value=guess_def,
                            step=0.01,
                            format="%.5f",
                            key=f"guess_{fit_id}_{pname}"
                        )
                        new_guesses.append(guess_val)
                    fit["param_guesses"] = new_guesses

                    # C) Subset data for this time range
                    sub_mask = (
                        (operated_data["Time"] >= fit["fit_start"]) &
                        (operated_data["Time"] <= fit["fit_end"])
                    )
                    sub_data = operated_data[sub_mask].copy()
                    if sub_data.empty:
                        st.warning("No data points in the selected time range.")
                    else:
                        st.write("#### Observed Data in this Fit Range")
                        plot_selected_wells(sub_data, ["Average"])

                        # D) Parse ODEs
                        ode_funcs = []
                        parse_failed = False
                        for expr in fit["odes"]:
                            f_callable = parse_ode(expr, fit["variables"], fit["parameters"])
                            if f_callable is None:
                                parse_failed = True
                                break
                            ode_funcs.append(f_callable)

                        if parse_failed:
                            st.error("One or more ODE expressions could not be parsed.")
                        else:
                            # define the ODE system
                            def ode_system(t, y, param_vec):
                                return [fun(t, y, param_vec) for fun in ode_funcs]

                            def cost_function(param_vec):
                                sol = solve_ivp(
                                    fun=lambda t, yy: ode_system(t, yy, param_vec),
                                    t_span=(sub_data["Time"].values[0], sub_data["Time"].values[-1]),
                                    y0=fit["initial_conditions"],
                                    t_eval=sub_data["Time"].values,
                                    method="RK45"
                                )
                                if not sol.success:
                                    return 1e10
                                x_model = sol.y[0]
                                x_obs = sub_data["Average"].values
                                return np.sum((x_obs - x_model)**2)

                            # E) Fit ODE button: try/except/else
                            fit_button_key = f"btn_fit_{fit_id}"
                            pressed_fit_button = st.button(f"Fit ODE (Fit {i+1})", key=fit_button_key)
                            if pressed_fit_button:
                                try:
                                    result = minimize(cost_function, x0=fit["param_guesses"], method="L-BFGS-B")
                                except Exception as e:
                                    st.error(f"Exception during optimization: {e}")
                                else:
                                    if not result.success:
                                        st.error(f"Optimization failed: {result.message}")
                                    else:
                                        fitted_params = result.x
                                        st.success(f"Fitted parameters: {dict(zip(fit['parameters'], fitted_params))}")

                                        sol_nom = solve_ivp(
                                            fun=lambda t, yy: ode_system(t, yy, fitted_params),
                                            t_span=(sub_data["Time"].values[0], sub_data["Time"].values[-1]),
                                            y0=fit["initial_conditions"],
                                            t_eval=sub_data["Time"].values,
                                            method="RK45"
                                        )
                                        if not sol_nom.success:
                                            st.warning(f"Solver failed after fitting: {sol_nom.message}")

                                        def residuals(p):
                                            tmp_sol = solve_ivp(
                                                fun=lambda t, yy: ode_system(t, yy, p),
                                                t_span=(sub_data["Time"].values[0], sub_data["Time"].values[-1]),
                                                y0=fit["initial_conditions"],
                                                t_eval=sub_data["Time"].values,
                                                method="RK45"
                                            )
                                            if not tmp_sol.success:
                                                return np.ones_like(sub_data["Average"].values)*1e5
                                            return sub_data["Average"].values - tmp_sol.y[0]

                                        r0 = residuals(fitted_params)
                                        N = len(r0)
                                        k_ = len(fitted_params)
                                        SSR = np.sum(r0**2)
                                        dof = N - k_
                                        sigma2 = SSR/dof if dof>0 else SSR

                                        J = np.zeros((N, k_))
                                        eps = 1e-6
                                        for j in range(k_):
                                            dp = np.zeros_like(fitted_params)
                                            dp[j] = eps
                                            r1 = residuals(fitted_params + dp)
                                            J[:, j] = (r1 - r0)/eps
                                        try:
                                            JTJ_inv = np.linalg.inv(J.T @ J)
                                            param_cov = sigma2*JTJ_inv
                                        except np.linalg.LinAlgError:
                                            param_cov = None
                                            st.warning("Jacobian is singular; cannot compute param covariance")

                                        AIC = 2*k_ + N*np.log(SSR/N)
                                        BIC = k_*np.log(N) + N*np.log(SSR/N)

                                        if param_cov is not None:
                                            std_errors = np.sqrt(np.diag(param_cov))
                                        else:
                                            std_errors = np.full(k_, np.nan)

                                        t_stats = fitted_params/std_errors
                                        pvals = 2*(1 - t_dist.cdf(np.abs(t_stats), df=dof))

                                        ci_results = compute_ode_ci_for_X(
                                            ode_system_func=ode_system,
                                            param_values=fitted_params,
                                            param_cov=param_cov,
                                            y0=fit["initial_conditions"],
                                            t_eval=sub_data["Time"].values,
                                            alpha=0.05
                                        )

                                        fit_result_dict = {
                                            "params": fitted_params,
                                            "param_cov": param_cov,
                                            "AIC": AIC,
                                            "BIC": BIC,
                                            "p_values": pvals,
                                            "solver_success": sol_nom.success,
                                            "time_eval": sub_data["Time"].values,
                                            "solution_y": sol_nom.y if sol_nom.success else None,
                                            "RSS": SSR,
                                        }
                                        # Update keys using compute_ode_ci_for_X output
                                        if ci_results is not None:
                                            fit_result_dict.update({
                                                "fit_time": ci_results["t"],
                                                "X_lower": ci_results["CI_lower"],
                                                "X_upper": ci_results["CI_upper"],
                                                "X_std": ci_results["std"]
                                            })
                                        fit["fit_results"] = fit_result_dict

                                        # Show summary table
                                        param_table = pd.DataFrame({
                                            "Parameter": fit["parameters"],
                                            "Estimate": fitted_params,
                                            "Std. Error": std_errors,
                                            "t-Statistic": t_stats,
                                            "p-Value": pvals
                                        })
                                        st.write("### Fitted Parameters & Stats")
                                        st.dataframe(param_table)

                                        stats_table = pd.DataFrame({
                                            "Metric": ["AIC", "BIC", "RSS"],
                                            "Value": [AIC, BIC, SSR]
                                        })
                                        st.dataframe(stats_table)

                                        st.info("Fit results saved in session_state!")
                                        if sol_nom.success and ci_results is not None:
                                            fig = go.Figure()
                                            fig.add_trace(go.Scatter(
                                                x=sub_data["Time"],
                                                y=sub_data["Average"],
                                                mode="markers",
                                                name="Observed X"
                                            ))
                                            fig.add_trace(go.Scatter(
                                                x=ci_results["t"],
                                                y=sol_nom.y[0],
                                                mode="lines",
                                                name="Fitted X"
                                            ))
                                            fig.add_trace(go.Scatter(
                                                x=np.concatenate([ci_results["t"], ci_results["t"][::-1]]),
                                                y=np.concatenate([ci_results["CI_upper"], ci_results["CI_lower"][::-1]]),
                                                fill="toself",
                                                fillcolor="rgba(173,216,230,0.3)",
                                                name="X 95% CI"
                                            ))
                                            if sol_nom.y.shape[0] > 1:
                                                fig.add_trace(go.Scatter(
                                                    x=ci_results["t"],
                                                    y=sol_nom.y[1],
                                                    mode="lines",
                                                    name="Fitted Y"
                                                ))
                                            fig.update_layout(
                                                title=f"ODE Fit (Fit {i+1})",
                                                xaxis_title="Time",
                                                yaxis_title="Value",
                                                template="plotly_white"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                            else:
                                # The user did NOT press "Fit ODE" this run.
                                # Re-display old results with full tables and plot if available.
                                old_res = fit.get("fit_results")
                                if old_res is not None:
                                    st.write("### Existing Fit Results from a Previous Run")
                                    # Reconstruct parameter table
                                    old_params = old_res.get("params", [])
                                    k_ = len(old_params)
                                    if old_res.get("param_cov") is not None:
                                        std_errors_old = np.sqrt(np.diag(old_res["param_cov"]))
                                    else:
                                        std_errors_old = np.full(k_, np.nan)
                                    param_table = pd.DataFrame({
                                        "Parameter": fit["parameters"],
                                        "Estimate": old_params,
                                        "Std. Error": std_errors_old,
                                        "p-Value": old_res.get("p_values", [np.nan]*k_)
                                    })
                                    st.dataframe(param_table)

                                    stats_table = pd.DataFrame({
                                        "Metric": ["AIC", "BIC", "RSS"],
                                        "Value": [
                                            old_res.get("AIC", np.nan),
                                            old_res.get("BIC", np.nan),
                                            old_res.get("RSS", np.nan)
                                        ]
                                    })
                                    st.dataframe(stats_table)

                                    x_time = old_res.get("fit_time")
                                    sol_y_old = old_res.get("solution_y")
                                    # Use .get() with a fallback (empty array) to prevent KeyError
                                    x_upper = old_res.get("X_upper")
                                    x_lower = old_res.get("X_lower")
                                    if x_time is not None and sol_y_old is not None and x_upper is not None and x_lower is not None:
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=old_res["time_eval"],
                                            y=sub_data["Average"],
                                            mode="markers",
                                            name="Observed X"
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=x_time,
                                            y=sol_y_old[0],
                                            mode="lines",
                                            name="Fitted X"
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=np.concatenate([x_time, x_time[::-1]]),
                                            y=np.concatenate([x_upper, x_lower[::-1]]),
                                            fill="toself",
                                            fillcolor="rgba(173,216,230,0.3)",
                                            name="X 95% CI"
                                        ))
                                        if sol_y_old.shape[0] > 1:
                                            fig.add_trace(go.Scatter(
                                                x=x_time,
                                                y=sol_y_old[1],
                                                mode="lines",
                                                name="Fitted Y"
                                            ))
                                        fig.update_layout(
                                            title=f"Old ODE Fit (Fit {i+1})",
                                            xaxis_title="Time",
                                            yaxis_title="Value",
                                            template="plotly_white"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Old results exist but are incomplete for plotting.")
                                else:
                                    st.info("No old results yet for this fit.")

                    # F) Delete Fit
                    if st.button(f"Delete ODE Fit {i+1}", key=f"delete_fit_{fit_id}"):
                        st.session_state["ode_fits"].pop(i)
                        st.success(f"Deleted ODE Fit {i+1}")
                        break  # re-run to update the list

        #----------------------------------------------------------------------
        # Summary Plot (optional)
        #----------------------------------------------------------------------
        st.subheader("Summary Plot of All ODE Fits (Optional)")
        if st.button("Generate Summary Plot for All ODE Fits"):
            fig_summary = plot_all_ode_fits_summary(
                ode_fits=st.session_state["ode_fits"],
                operated_data=operated_data,
                selected_operated_wells=selected_operated_wells
            )
            if isinstance(fig_summary, go.Figure):
                st.plotly_chart(fig_summary, use_container_width=True)
            else:
                st.error("Summary plot did not return a valid Plotly figure.")





    # Tab 6: Automatic Phase Detection
    # Inside the main() function, within Tab 6
    with tab6:
        st.header("🔍 Automatic Phase Detection (Merged)")

        # Retrieve your operated data and selected wells
        operated_data = st.session_state.get("operated_data")
        selected_operated_wells = st.session_state.get("selected_operated_wells", [])

        # Check for valid data
        if operated_data is None or not selected_operated_wells:
            st.warning("⚠️ Please perform operations on the data in the previous tabs before phase detection.")
        else:
            # Ensure we have an "Average" column
            if "Average" not in operated_data.columns:
                operated_data["Average"] = operated_data[selected_operated_wells].mean(axis=1)

            # Let user select which detection method
            method = st.selectbox(
                "Select Phase Detection Method",
                ["thresholded_derivative", "slope_based"],
                format_func=lambda m: "Thresholded Derivative" if m == "thresholded_derivative" else "Slope-Based"
            )

            # Common data references
            time = operated_data["Time"].values
            od_values = operated_data["Average"].values

            # Show method-specific input widgets
            if method == "thresholded_derivative":
                st.subheader("Thresholded Derivative Parameters")
                smoothing_window = st.slider("Smoothing Window", 3, 51, 5, step=2)
                polyorder = st.slider("Polynomial Order (Savitzky-Golay)", 1, 5, 2)
                derivative_threshold = st.number_input("Derivative Threshold", value=0.005, step=0.001,  format="%.5f")
                min_distance = st.slider("Min Distance Between Breakpoints", 1, 50, 5)

                # Button to run detection
                if st.button("Detect Phases"):
                    # Call your helper function (defined elsewhere)
                    phases, breakpoints, extra = detect_phases(
                        time=time,
                        od_values=od_values,
                        method="thresholded_derivative",
                        smoothing_window=smoothing_window,
                        polyorder=polyorder,
                        derivative_threshold=derivative_threshold,
                        min_distance=min_distance
                    )

                    if not phases:
                        st.error("No phases detected. Try adjusting parameters.")
                    else:
                        st.success(f"Detected {len(phases)} phase(s).")
                        # Plot using your helper function (defined elsewhere)
                        plot_detected_phases(
                            time, 
                            od_values, 
                            phases, 
                            derivative=extra.get("derivative"), 
                            slopes=None, 
                            change_points=breakpoints
                        )

                        # Display table of phases
                        st.subheader("Detected Phases")
                        phase_table = []
                        for idx, (start, end) in enumerate(phases):
                            phase_table.append({
                                "Phase": idx + 1,
                                "Start": start,
                                "End": end,
                                "Duration": end - start
                            })
                        st.dataframe(phase_table)

            else:  # slope_based
                st.subheader("Slope-Based Parameters")
                slope_window = st.slider("Slope Window Size", 3, 51, 5, step=2)
                slope_threshold = st.number_input("Slope Threshold", value=0.001, step=0.0001,  format="%.5f")

                if st.button("Detect Phases"):
                    phases, breakpoints, extra = detect_phases(
                        time=time,
                        od_values=od_values,
                        method="slope_based",
                        slope_window=slope_window,
                        slope_threshold=slope_threshold
                    )

                    if not phases:
                        st.error("No phases detected. Try adjusting parameters.")
                    else:
                        st.success(f"Detected {len(phases)} phase(s).")
                        # Plot
                        plot_detected_phases(
                            time,
                            od_values,
                            phases,
                            derivative=None,
                            slopes=extra.get("slopes"),
                            change_points=breakpoints
                        )

                        # Display table of phases
                        st.subheader("Detected Phases")
                        phase_table = []
                        for idx, (start, end) in enumerate(phases):
                            phase_table.append({
                                "Phase": idx + 1,
                                "Start": start,
                                "End": end,
                                "Duration": end - start
                            })
                        st.dataframe(phase_table)


    # Tab 7: Growth Models
    with tab7:
        st.title("Bacterial Growth Models")

        # Adding the image with controlled dimensions
        image_path = os.path.join("assests", "f1.png")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            # Streamlit handles scaling within the container
            st.image(image, caption="Example: Growth Model Visualization", width=600)  # Specify the desired width
        else:
            st.warning("Image file 'f1.png' not found. Please ensure it is placed in the correct directory.")

        st.write("## Exponential Growth Model")
        st.write("The exponential growth model is described by the following equations:")
        st.latex(r'''
            \frac{dX}{dt} = \mu X
        ''')
        st.latex(r'''
            x(t) = X_0 e^{\mu t}
        ''')
        st.code("""
    def exponential_growth(t, mu, X0):
        return X0 * np.exp(mu * t)
        """)
        st.write(" where x0 is the initial bacterial biomass at time 0, mu is the growth rate")

        st.write("## Logistic Growth Model")
        st.write("The logistic growth model with saturation is described by the following equations:")
        st.latex(r'''
            \frac{dX}{dt} = \mu X \left(1 - \frac{X}{K}\right)
        ''')
        st.latex(r'''
            x(t) = \frac{X_0 e^{\mu t}}{1 + \frac{X_0}{K} \left(e^{\mu t} - 1\right)}
        ''')
        st.write("where X is the biomass, X0 is the initial biomass at time 0, mu is the growth rate, K is the saturation constant (maximal OD)")
        st.code("""
    def logistic_growth(t, mu, X0, K):
        return (X0 * np.exp(mu * t)) / (1 + (X0 / K) * (np.exp(mu * t) - 1))
        """)

        st.write("## Baranyi Model")
        st.write("The Baranyi model for lag-exponential growth is described by the following equations:")
        st.latex(r'''
            \frac{dX}{dt} = \mu \frac{q(t)}{1 + q(t)} X
        ''')
        st.latex(r'''
            \frac{dq}{dt} = \mu q
        ''')
        st.latex(r'''
            x(t) = X_0 \frac{1 + q_0 e^{\mu t}}{1 + q_0}
        ''')
        st.code("""
    def baranyi_growth(t, X0, mu, q0):
        q_t = q0 * np.exp(mu * t)
        return X0 * (1 + q_t) / (1 + q0)
        """)
        st.write("where x0 is the initial biomass at time 0, mu is the growth rate, q0 is a physiological state of the cell in constant environment (for example the enzymes that need to accumulate to adapt to the new condition).")

        st.write("## Lag-Exponential-Saturation Growth Model")
        st.write("The lag-exponential-saturation growth model is described by the following equations:")
        st.latex(r'''
            \frac{dX}{dt} = \mu \frac{q(t)}{1 + q(t)} X \left(1 - \frac{X}{K}\right)
        ''')
        st.latex(r'''
            \frac{dq}{dt} = \mu q
        ''')
        st.latex(r'''
            x(t) = X_0 \frac{1 + q_0 e^{\mu t}}{1 + q_0 - q_0 \frac{X_0}{K} + \frac{q_0 X_0}{K} e^{\mu t}}
        ''')
        st.code("""
    def lag_exponential_saturation_growth(t, mu, X0, q0, K):
        return X0 * (1 + q0 * np.exp(mu * t)) / (1 + q0 - q0 * (X0 / K) + (q0 * X0 / K) * np.exp(mu * t))
        """)
        st.write("where x0 is the initial biomass at time 0, K is the saturation constant (maximal OD), mu is the growth rate, q0 is a physiological state of the cell in constant environment (for example the enzymes that need to accumulate to adapt to the new condition).")
        st.write("X and X0 are usually measured in [a.u.] at OD600 or in concentration units, K has the same units of the biomass, mu is measured in [1/t], q0  is adimensional.")




if __name__ == "__main__":
    main()