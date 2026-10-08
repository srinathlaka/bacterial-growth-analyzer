import streamlit as st
import numpy as np
import pandas as pd
import json
import uuid
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

from utils.plotting import (
    CHART_CONFIG,
    plot_avg_sd_operated,
    plot_selected_wells,
    plot_confidence_intervals,
    plot_fitted_curves,
    plot_phase_fit_with_ci
)
from utils.file_io import create_custom_model
from utils.fitting import compute_confidence_intervals, compute_lag_time, compute_fit_statistics
from utils.models import (
    MODEL_PARAMS,
    MODEL_FUNCTIONS,
    default_guesses,
    PARAMETER_UNITS
)

def display_tab_fitting():
    st.header("Growth Model Fitting")
    
    # Check if background subtraction was skipped. Trust the data over the flag:
    # if subtracted data exists, the flag is stale and the warning would be wrong.
    has_bg_subtracted_data = any(
        key.endswith("_bg_subtracted")
        for key in st.session_state.get("groups_data", {})
    )
    if st.session_state.get("bg_subtraction_skipped", False) and not has_bg_subtracted_data:
        st.warning("""
        ⚠️ **Background subtraction was skipped**
        
        You're analyzing raw data without background correction.
        This may affect growth parameter estimation if your data has significant background signal.
        """)
    
    st.subheader("Fitting")

    # Make sure we have "operated_data"
    if "operated_data" not in st.session_state or st.session_state["operated_data"] is None:
        # fallback to single-group data
        groups_data = st.session_state.get("groups_data", {})
        if st.session_state.get("num_groups", 1) == 1 and "Group_1_bg_subtracted" in groups_data:
            st.session_state["operated_data"] = groups_data.get("Group_1_bg_subtracted")
        else:
            st.error("Operated data not found. Complete Tab 2 and Tab 3 first.")
            return

    operated_data = st.session_state["operated_data"]
    selected_operated_wells = st.session_state.get("selected_operated_wells", [])

    if operated_data is None or not selected_operated_wells:
        st.warning("Operated data or selected wells are missing. Check previous tabs.")
        return

    # Keep only wells the data actually has. The selection can change after the
    # table was built, which would otherwise raise a bare KeyError here.
    missing_wells = [w for w in selected_operated_wells if w not in operated_data.columns]
    if missing_wells:
        st.warning(
            f"⚠️ These selected wells are not in the prepared data: {', '.join(missing_wells)}. "
            f"They were selected after the data was prepared. Return to the Background tab "
            f"to rebuild it, or continue with the wells that are present."
        )
        selected_operated_wells = [w for w in selected_operated_wells if w in operated_data.columns]
        if not selected_operated_wells:
            st.error("None of the selected wells are present in the prepared data.")
            return
        st.session_state["selected_operated_wells"] = selected_operated_wells

    if "Average" not in operated_data.columns:
        operated_data["Average"] = operated_data[selected_operated_wells].mean(axis=1)

    st.info("Operated Data Preview:")
    st.dataframe(operated_data.head())

    st.subheader("Average and Standard Deviation of Operated Data")
    try:
        plot_avg_sd_operated(operated_data, selected_operated_wells)
    except Exception as e:
        st.error(f"Error plotting operated data: {e}")

    # Automatic Fitting
    st.markdown("### Automatic Fitting via JSON Configuration")
    uploaded_json_fit = st.file_uploader("Upload JSON for Automatic Fitting", type=["json"], key="tab4_json_upload")
    if uploaded_json_fit is not None:
        _handle_automatic_fits(uploaded_json_fit, operated_data, selected_operated_wells)
    else:
        st.info("No JSON configuration uploaded. You can add fits manually below.")

    # Manual Fitting UI
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

    # Display & edit existing fits
    _display_existing_phases(operated_data, selected_operated_wells)

    # Summary Plot
    st.markdown("### Summary of All Fit Results")
    if st.session_state.get("phases"):
        fitted_phases = [
            phase for phase in st.session_state["phases"]
            if phase.get("fit_results") is not None
        ]
        if fitted_phases:
            # Generate a stable key based on the number of phases and their IDs
            phase_ids = "_".join([str(phase.get("id", ""))[:4] for phase in fitted_phases][:3])
            summary_key = f"summary_plot_{len(fitted_phases)}_{phase_ids}"
            
            # Create the summary plot
            fig_summary = plot_phase_fit_with_ci(fitted_phases, operated_data, selected_operated_wells)
            
            # Use the stable key for the plot
            st.plotly_chart(fig_summary, theme=None, config=CHART_CONFIG, use_container_width=True, key=summary_key)

            # Tidy results table + CSV export
            st.markdown("#### Fitted Parameters (All Fits)")
            results_table = build_results_table(fitted_phases)
            if not results_table.empty:
                st.dataframe(results_table, key="all_fit_results_table")
                st.download_button(
                    label="📥 Download Results CSV",
                    data=results_table.to_csv(index=False).encode("utf-8"),
                    file_name="fit_results.csv",
                    mime="text/csv",
                    key="download_fit_results_csv"
                )
        else:
            st.info("No fits available to generate a summary plot.")

        # Build JSON for all fits
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
        st.download_button(
            label="💾 Download Fit Configuration JSON",
            data=edited_json,
            file_name="fit_configuration.json",
            mime="application/json",
            key="download_fit_config_json"
        )
    else:
        st.info("No fit results to summarize.")


def _handle_automatic_fits(uploaded_json_fit, operated_data, selected_wells):
    try:
        fit_config = json.load(uploaded_json_fit)
        auto_fits = fit_config.get("fit_configuration", [])
        if not auto_fits:
            st.warning("No fit configurations found in the JSON file.")
            return
        for idx, fit_conf in enumerate(auto_fits):
            st.markdown(f"#### Processing Fit {idx+1}")
            time_interval = fit_conf.get("time_interval", {})
            start = time_interval.get("start")
            end = time_interval.get("end")
            model = fit_conf.get("model")
            initial_guesses = fit_conf.get("initial_guesses")
            bounds = fit_conf.get("bounds")
            custom_model_expr = fit_conf.get("custom_model_expr")
            custom_params = fit_conf.get("custom_params")
            wells = fit_conf.get("selected_operated_wells", selected_wells)

            if start is None or end is None or model is None or initial_guesses is None:
                st.error("Missing required fields: time_interval, model, initial_guesses.")
                continue

            phase_data = operated_data[(operated_data["Time"]>=start) & (operated_data["Time"]<=end)][["Time"]+wells].copy()
            phase_data = phase_data.dropna(subset=wells)
            if phase_data.empty:
                st.warning(f"No data points in interval {start} to {end}.")
                continue
            phase_data.loc[:, "Average"] = phase_data[wells].mean(axis=1)
            time_vals = phase_data["Time"].values
            y_data = phase_data["Average"].values

            from utils.file_io import create_custom_model
            from utils.models import MODEL_FUNCTIONS, MODEL_PARAMS

            if model == "Custom Function":
                model_func = create_custom_model(custom_model_expr, custom_params)
                param_names = [p.strip() for p in custom_params if p.strip()]
            else:
                model_func = MODEL_FUNCTIONS.get(model)
                param_names = MODEL_PARAMS.get(model,[])
            if model_func is None:
                st.error(f"No model function available for model {model}.")
                continue

            from scipy.optimize import curve_fit

            try:
                if bounds:
                    lower_bounds = bounds.get("lower")
                    upper_bounds = bounds.get("upper")
                    popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
                else:
                    popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses)
            except Exception as e_fit:
                st.error(f"Fitting failed for {start}-{end}: {e_fit}")
                continue

            stats = compute_fit_statistics(time_vals, y_data, popt, pcov, model_func, param_names)
            if stats is None:
                st.error(
                    f"Not enough data points ({len(y_data)}) to fit {len(popt)} parameters "
                    f"in interval {start}-{end}. Widen the time window or choose a simpler model."
                )
                continue

            y_pred = stats["y_pred"]
            perr = stats["param_errors"]
            t_statistic = stats["t_statistic"]
            p_values = stats["p_values"]
            AIC = stats["AIC"]
            R_squared = stats["R_squared"]
            variance_ratio = stats["variance_ratio"]
            lower_bound_ci = stats["lower_bound"]
            upper_bound_ci = stats["upper_bound"]

            if stats["poor_fit"]:
                st.warning(f"⚠️ Model fit quality is questionable (R² = {R_squared:.4f}, Variance ratio = {variance_ratio:.4f})")

            phase_dict = {
                "id": str(uuid.uuid4()),
                "phase": idx+1,
                "time_interval": {"start":start, "end":end},
                "selected_operated_wells": wells,
                "model": model,
                "parameters": param_names,
                "initial_guesses": initial_guesses,
                "bounds": bounds
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

            lag_time = stats["lag_time"]
            lag_time_std_err = stats["lag_time_std_err"]
            if lag_time is not None:
                fit_results["lag_time"] = lag_time
                fit_results["lag_time_std_err"] = lag_time_std_err
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
            st.success(f"Fit {idx+1} completed for interval {start}-{end} using model {model}.")
            plot_fitted_curves(phase_data, time_vals, y_data, y_pred, model)
            plot_confidence_intervals(phase_data, lower_bound_ci, upper_bound_ci, y_pred, phase_data[wells].std(axis=1))

            parameter_labels = [f"{p}" for p in param_names]
            param_table = pd.DataFrame({
                "Parameter": parameter_labels,
                "Estimate": [format_scientific(val) for val in popt],
                "Std. Error": [format_scientific(val) for val in perr],
                "t-Statistic": [format_scientific(val) for val in t_statistic],
                "p-Value": [format_scientific(p) for p in p_values]
            })

            if lag_time is not None:
                lag_row = pd.DataFrame([{
                    "Parameter": f"lag_time {PARAMETER_UNITS.get('lag_time', '[time]')}",
                    "Estimate": format_scientific(lag_time) if lag_time is not None else "N/A",
                    "Std. Error": format_scientific(lag_time_std_err) if lag_time_std_err is not None else "N/A",
                    "t-Statistic": "N/A",
                    "p-Value": "N/A"
                }])
                param_table = pd.concat([param_table, lag_row], ignore_index=True)

            st.dataframe(param_table)
    except Exception as e:
        st.error(f"Error processing JSON file: {e}")


def _display_existing_phases(operated_data, selected_operated_wells):
    from utils.plotting import plot_fitted_curves, plot_confidence_intervals
    from utils.models import MODEL_FUNCTIONS, MODEL_PARAMS, default_guesses
    from utils.fitting import compute_fit_statistics
    from scipy.optimize import curve_fit

    for i, phase in enumerate(st.session_state.get("phases", [])):
        if "id" not in phase:
            phase["id"] = str(uuid.uuid4())
        phase_id = phase["id"]
        with st.expander(f"Manual Fit {i+1}"):
            try:
                t_interval = phase.get("time_interval", {})
                data_min = float(operated_data['Time'].min())
                data_max = float(operated_data['Time'].max())

                def _clamp(value, fallback):
                    # A phase loaded from JSON may fall outside the current data
                    # range; number_input rejects a value outside min/max.
                    try:
                        return min(max(float(value), data_min), data_max)
                    except (TypeError, ValueError):
                        return fallback

                fit_start = st.number_input(
                    f"Start Time for Fit {i+1}",
                    min_value=data_min,
                    max_value=data_max,
                    value=_clamp(t_interval.get("start", data_min), data_min),
                    step=0.1,
                    format="%.5f",
                    key=f"start_{i}"
                )
                fit_end = st.number_input(
                    f"End Time for Fit {i+1}",
                    min_value=data_min,
                    max_value=data_max,
                    value=_clamp(t_interval.get("end", data_max), data_max),
                    step=0.1,
                    format="%.5f",
                    key=f"end_{i}"
                )
                phase["time_interval"] = {"start": float(fit_start), "end": float(fit_end)}

                # Delete Fit
                if st.button(f"Delete Fit {i+1}", key=f"delete_phase_{phase_id}"):
                    st.session_state["phases"].pop(i)
                    st.success(f"Deleted Fit {i+1}")
                    return

                phase_data = operated_data[(operated_data["Time"]>=float(fit_start)) &
                                           (operated_data["Time"]<=float(fit_end))].copy()
                if phase_data.empty:
                    st.warning("No data points in this interval.")
                    continue
                phase_data["Average"] = phase_data[selected_operated_wells].mean(axis=1)
                time_vals = phase_data["Time"].values
                y_data = phase_data["Average"].values

                # First, store the model selection in session state
                if f"model_selection_{i}" not in st.session_state:
                    st.session_state[f"model_selection_{i}"] = phase.get("model", "Exponential Growth")

                # Then use the session state value to set the default index
                model_options = [
                    "Exponential Growth",
                    "Logistic Growth",
                    "Baranyi Growth",
                    "Lag-Exponential-Saturation Growth",
                    "Gompertz Growth",
                    "Custom Function",
                    "Automatic Fit"
                ]
                default_index = model_options.index(st.session_state[f"model_selection_{i}"]) if st.session_state[f"model_selection_{i}"] in model_options else 0

                # Use index parameter to maintain selection after rerun
                phase["model"] = st.selectbox(
                    f"Select Model for Fit {i+1}",
                    model_options,
                    index=default_index,
                    key=f"model_{i}"
                )

                # Update the session state when selection changes
                st.session_state[f"model_selection_{i}"] = phase["model"]

                if phase["model"] == "Custom Function":
                    # Store initial values in session state to preserve them
                    if f"custom_model_expr_{i}" not in st.session_state:
                        st.session_state[f"custom_model_expr_{i}"] = phase.get("custom_model_expr", "X*exp(mu*t)")
                    
                    if f"custom_params_{i}" not in st.session_state:
                        params_str = ", ".join(phase.get("custom_params", [])) or "X, mu"
                        st.session_state[f"custom_params_{i}"] = params_str
                    
                    # Add helpful examples
                    st.info("""
                    **Custom Function Examples:**
                    - `X0*exp(mu*t)`: Exponential growth
                    - `K/(1 + ((K-X0)/X0)*exp(-mu*t))`: Logistic growth
                    - `A*(1-exp(-exp(mu*e/A*(lambda-t)+1)))`: Gompertz model
                    
                    Use `t` as the independent variable (time).
                    """)
                    
                    # Use the stored values as defaults with better column layout
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        phase["custom_model_expr"] = st.text_input(
                            f"Custom Model Expression for Fit {i+1}",
                            value=st.session_state[f"custom_model_expr_{i}"],
                            key=f"custom_model_input_{i}"
                        )
                    with col2:
                        st.write("Parameters must be in expression!")
                    
                    # Update session state immediately
                    st.session_state[f"custom_model_expr_{i}"] = phase["custom_model_expr"]
                    
                    params_str = st.text_input(
                        f"Parameters to Optimize (comma-separated) for Fit {i+1}",
                        value=st.session_state[f"custom_params_{i}"],
                        help="Example: X0, mu, K (must match variables in expression)",
                        key=f"custom_params_input_{i}"
                    )
                    # Update session state immediately
                    st.session_state[f"custom_params_{i}"] = params_str
                    phase["custom_params"] = [p.strip() for p in params_str.split(",") if p.strip()]
                    
                    try:
                        # Validate parameters match expression
                        for param in phase["custom_params"]:
                            if param.strip() not in phase["custom_model_expr"] and param.strip() != "t":
                                st.warning(f"Parameter '{param}' not found in expression. Did you misspell it?")
                        
                        from utils.file_io import create_custom_model
                        model_func = create_custom_model(phase["custom_model_expr"], phase["custom_params"])
                        phase["parameters"] = phase["custom_params"]
                        
                        # Show a preview of the function with test values
                        test_t = 1.0
                        test_params = [1.0] * len(phase["custom_params"])
                        try:
                            test_result = model_func(test_t, *test_params)
                            st.success(f"✅ Custom function parsed successfully! f(t=1) = {test_result:.4f}")
                        except Exception as e_test:
                            st.warning(f"Function parsed but test evaluation failed: {e_test}")
                    except Exception as e:
                        st.error(f"Error in custom function: {str(e)}")
                        # Don't clear the inputs on error - they're already preserved in session state
                        phase["parameters"] = phase["custom_params"]
                elif phase["model"] == "Automatic Fit":
                    st.info("""
                    **What is AIC?**
                    
                    When using Automatic Fit, the software compares several candidate models and selects the one with the lowest AIC (Akaike Information Criterion).
                    
                    **AIC** is a statistical measure that balances model fit and complexity. It rewards models that fit the data well, but penalizes models that use too many parameters (to avoid overfitting). Lower AIC values indicate a better model for your data among the candidates tested.
                    """)
                    st.info("Automatic Fit: evaluating candidate models.")
                    best_model, best_popt, best_pcov, best_aic, best_candidate = None,None,None,np.inf,None
                    from utils.models import MODEL_FUNCTIONS, MODEL_PARAMS, default_guesses

                    # Add Gompertz Growth to the list of candidate models
                    for candidate in [m for m in MODEL_FUNCTIONS.keys() if m not in ["Custom Function", "Automatic Fit", "Power Law", "Polynomial Function"]]:
                        try:
                            cf_model_func = MODEL_FUNCTIONS[candidate]
                            
                            # Use smart default guesses with proper function calls
                            if candidate in default_guesses:
                                # Call the lambda with both y_data and time_vals
                                if callable(default_guesses[candidate]):
                                    guesses = default_guesses[candidate](y_data, time_vals)
                                else:
                                    guesses = [1.0] * len(MODEL_PARAMS.get(candidate, []))
                                
                            popt_candidate, pcov_candidate = curve_fit(cf_model_func, time_vals, y_data, p0=guesses)
                            y_pred_candidate = cf_model_func(time_vals, *popt_candidate)
                            rss_candidate = np.sum((y_data - y_pred_candidate)**2)
                            dof_candidate = len(y_data)-len(popt_candidate)
                            aic_candidate = 2*len(popt_candidate) + len(y_data)*np.log(rss_candidate/len(y_data))
                            st.write(f"Candidate {candidate}: AIC = {aic_candidate:.2f}")
                            if aic_candidate<best_aic:
                                best_aic=aic_candidate
                                best_model=cf_model_func
                                best_popt=popt_candidate
                                best_pcov=pcov_candidate
                                best_candidate=candidate
                        except Exception as e_cand:
                            st.warning(f"Candidate {candidate} failed: {e_cand}")
                            continue
                    if best_model is None:
                        st.error("Automatic Fit could not find a suitable model.")
                        continue
                    st.success(f"Automatic Fit selected: {best_candidate} (AIC={best_aic:.2f})")
                    model_func=best_model
                    phase["model"]=best_candidate
                    from utils.models import MODEL_PARAMS
                    phase["parameters"]=MODEL_PARAMS.get(best_candidate,[])
                    popt=best_popt
                    pcov=best_pcov
                    y_pred=model_func(time_vals,*popt)
                else:
                    from utils.models import MODEL_FUNCTIONS, MODEL_PARAMS
                    model_func=MODEL_FUNCTIONS.get(phase["model"])
                    phase["parameters"]=MODEL_PARAMS.get(phase["model"],[])

                if model_func is None:
                    st.error("No model function available.")
                    continue

                st.markdown("#### Enter Initial Guesses and Optional Bounds")
                params_list=phase.get("parameters",[])
                initial_guesses_list=[]
                lower_bounds=[]
                upper_bounds=[]
                use_bounds=False

                for param in params_list:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        # Initialize default_value for each parameter
                        default_value = 1.0  # Or whatever appropriate default you want
                        # Get smart defaults using the lambda functions
                        if phase["model"] in default_guesses:
                            param_index = params_list.index(param)
                            try:
                                # Call lambda with both y_data and time_vals
                                if callable(default_guesses[phase["model"]]):
                                    defaults = default_guesses[phase["model"]](y_data, time_vals)
                                    default_value = defaults[param_index] if param_index < len(defaults) else 1.0
                                else:
                                    default_value = default_guesses[phase["model"]][param_index]
                            except Exception:
                                default_value = 1.0

                        # Backstop: number_input rejects inf/nan outright, which
                        # would abort the whole fit panel with a cryptic error.
                        if not np.isfinite(default_value):
                            default_value = 1.0

                        guess = st.number_input(f"Initial guess for {param}",
                                               value=float(default_value),
                                               step=0.01, 
                                               format="%.5f", 
                                               key=f"{param}_{i}_guess")
                        initial_guesses_list.append(guess)
                    with col_b:
                        bound_flag = st.checkbox(f"Use bounds for {param}?", value=False, key=f"use_bounds_{param}_{i}")
                    if bound_flag:
                        use_bounds=True
                        with col_c:
                            lb=st.number_input(f"Lower bound for {param}", value=0.0, key=f"lb_{param}_{i}", format="%.5f")
                            lower_bounds.append(lb)
                        with col_d:
                            ub=st.number_input(f"Upper bound for {param}", value=10.0, key=f"ub_{param}_{i}", format="%.5f")
                            upper_bounds.append(ub)
                    else:
                        lower_bounds.append(-np.inf)
                        upper_bounds.append(np.inf)

                phase["initial_guesses"]=initial_guesses_list
                phase["bounds"]={"lower":lower_bounds,"upper":upper_bounds} if use_bounds else None

                if st.button(f"Fit Model for Fit {i+1}", key=f"fit_model_{i}"):
                    try:
                        if phase["bounds"]:
                            popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses_list, bounds=(lower_bounds, upper_bounds))
                        else:
                            popt, pcov = curve_fit(model_func, time_vals, y_data, p0=initial_guesses_list)
                        stats = compute_fit_statistics(
                            time_vals, y_data, popt, pcov, model_func, params_list
                        )
                        if stats is None:
                            st.error(
                                f"Not enough data points ({len(y_data)}) to fit {len(popt)} parameters "
                                f"in this interval. Widen the time window or choose a simpler model."
                            )
                            continue

                        y_pred = stats["y_pred"]
                        perr = stats["param_errors"]
                        t_statistic = stats["t_statistic"]
                        p_values = stats["p_values"]
                        AIC = stats["AIC"]
                        R_squared = stats["R_squared"]
                        residual_variance = stats["residual_variance"]
                        variance_ratio = stats["variance_ratio"]
                        lower_bound_ci = stats["lower_bound"]
                        upper_bound_ci = stats["upper_bound"]

                        st.write(f"Goodness of fit: R² = {R_squared:.4f}")
                        st.write(f"Residual variance: {residual_variance:.8f}")

                        if stats["poor_fit"]:
                            st.warning(f"⚠️ Model fit quality is questionable (R² = {R_squared:.4f}, Variance ratio = {variance_ratio:.4f})")
                            st.error(f"""
                            ⚠️ WARNING: THIS MODEL DOES NOT FIT THE DATA WELL
                            - R² = {R_squared:.4f} (should be > 0.90)
                            - Variance ratio = {variance_ratio:.4f} (should be < 0.20)
                            
                            The p-values shown below are ADJUSTED to reflect the poor fit quality
                            and should NOT be interpreted as statistically significant.
                            """)

                        phase["fit_results"]={
                            "phase_time":time_vals,
                            "fit":y_pred,
                            "lower_bound":lower_bound_ci,
                            "upper_bound":upper_bound_ci,
                            "std_dev":phase_data[selected_operated_wells].std(axis=1).values,
                            "parameters":popt,
                            "param_errors":perr,
                            "AIC":AIC,
                            "t_statistic":t_statistic,
                            "p_values":p_values
                        }

                        lag_time = stats["lag_time"]
                        lag_time_std_err = stats["lag_time_std_err"]
                        if lag_time is not None:
                            phase["fit_results"]["lag_time"] = lag_time
                            phase["fit_results"]["lag_time_std_err"] = lag_time_std_err
                        phase.setdefault("phase", i+1)
                        phase["phase_time"]=time_vals
                        phase["fit"]=y_pred
                        phase["lower_bound"]=lower_bound_ci
                        phase["upper_bound"]=upper_bound_ci
                        phase["std_dev"]=phase_data[selected_operated_wells].std(axis=1).values
                        phase["parameters"]=popt
                        phase["param_errors"]=perr
                        phase["AIC"]=AIC
                        phase["t_statistic"]=t_statistic
                        phase["p_values"]=p_values

                        st.success("Model fitted successfully!")
                        plot_fitted_curves(phase_data, time_vals, y_data, y_pred, phase["model"], phase["id"])
                        plot_confidence_intervals(phase_data, lower_bound_ci, upper_bound_ci, y_pred, phase_data[selected_operated_wells].std(axis=1), phase["id"])

                        model_name = phase.get("model", "Unknown Model")
                        param_names = MODEL_PARAMS.get(model_name, [])

                        # Create better parameter labels with units
                        param_labels = []
                        for i, param in enumerate(phase.get("parameters", [])):
                            param_name = param_names[i] if i < len(param_names) else f"Param {i+1}"
                            unit = PARAMETER_UNITS.get(param_name, "")
                            param_labels.append(f"{param_name} {unit}")

                        param_table=pd.DataFrame({
                            "Parameter": param_labels,
                            "Estimate": [format_scientific(val) for val in popt],
                            "Std. Error": [format_scientific(val) for val in perr],
                            "t-Statistic": [format_scientific(val) for val in t_statistic],
                            "p-Value": [format_scientific(p) for p in p_values],
                            "Fit Quality": ["POOR" if variance_ratio > 0.2 or R_squared < 0.90 else "GOOD" for _ in popt],
                            "R²": [R_squared for _ in popt],  # Add R² for context
                            "Variance Ratio": [variance_ratio for _ in popt]  # Add variance ratio for context
                        })

                        if lag_time is not None:
                            lag_row = pd.DataFrame([{
                                "Parameter": f"lag_time {PARAMETER_UNITS.get('lag_time', '[time]')}",
                                "Estimate": format_scientific(lag_time) if lag_time is not None else "N/A",
                                "Std. Error": format_scientific(lag_time_std_err) if lag_time_std_err is not None else "N/A",
                                "t-Statistic": "N/A",
                                "p-Value": "N/A",
                                "Fit Quality": "POOR" if variance_ratio > 0.2 or R_squared < 0.90 else "GOOD",
                                "R²": R_squared,
                                "Variance Ratio": variance_ratio
                            }])
                            param_table = pd.concat([param_table, lag_row], ignore_index=True)

                        st.dataframe(param_table, key=f"param_table_{phase['id']}")
                    except Exception as e_fit:
                        st.error(f"Error fitting model for Fit {i+1}: {e_fit}")
            except Exception as e_manual:
                st.error(f"Error in manual fitting for Fit {i+1}: {e_manual}")

def build_results_table(fitted_phases):
    """
    Flatten every fitted phase into one tidy table: a row per parameter per fit,
    plus a lag_time row where applicable. Suitable for CSV export/reporting.
    """
    rows = []
    for phase in fitted_phases:
        fit_results = phase.get("fit_results") or {}
        model_name = phase.get("model", "Unknown")
        interval = phase.get("time_interval", {}) or {}
        param_names = MODEL_PARAMS.get(model_name) or phase.get("custom_params") or []
        estimates = fit_results.get("parameters", [])
        errors = fit_results.get("param_errors", [])
        t_stats = fit_results.get("t_statistic", [])
        p_vals = fit_results.get("p_values", [])

        for idx, estimate in enumerate(estimates):
            name = param_names[idx] if idx < len(param_names) else f"Param {idx+1}"
            rows.append({
                "Fit": phase.get("phase", ""),
                "Model": model_name,
                "Start Time": interval.get("start"),
                "End Time": interval.get("end"),
                "Parameter": name,
                "Unit": PARAMETER_UNITS.get(name, ""),
                "Estimate": estimate,
                "Std. Error": errors[idx] if idx < len(errors) else None,
                "t-Statistic": t_stats[idx] if idx < len(t_stats) else None,
                "p-Value": p_vals[idx] if idx < len(p_vals) else None,
                "AIC": fit_results.get("AIC"),
            })

        if fit_results.get("lag_time") is not None:
            rows.append({
                "Fit": phase.get("phase", ""),
                "Model": model_name,
                "Start Time": interval.get("start"),
                "End Time": interval.get("end"),
                "Parameter": "lag_time",
                "Unit": PARAMETER_UNITS.get("lag_time", "[time]"),
                "Estimate": fit_results.get("lag_time"),
                "Std. Error": fit_results.get("lag_time_std_err"),
                "t-Statistic": None,
                "p-Value": None,
                "AIC": fit_results.get("AIC"),
            })

    return pd.DataFrame(rows)


def format_scientific(value, precision=6):
    """
    Format value in scientific notation with appropriate precision based on magnitude.
    """
    try:
        float_val = float(value)
        abs_val = abs(float_val)
        if abs_val < 0.001 or abs_val > 1000:
            return f"{float_val:.{precision}e}"
        else:
            return f"{float_val:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{float_val:.{precision}f}" else f"{float_val}"
    except Exception:
        return str(value)
