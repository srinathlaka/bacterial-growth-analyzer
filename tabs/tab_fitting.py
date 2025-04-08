import streamlit as st
import numpy as np
import pandas as pd
import json
import uuid
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

from utils.plotting import (
    plot_avg_sd_operated,
    plot_selected_wells,
    plot_confidence_intervals,
    plot_fitted_curves,
    plot_phase_fit_with_ci
)
from utils.file_io import create_custom_model
from utils.fitting import compute_confidence_intervals
from utils.models import (
    MODEL_PARAMS,
    MODEL_FUNCTIONS,
    default_guesses,
    PARAMETER_UNITS
)

def display_tab_fitting():
    """
    Tab 4: Fitting. 
    - Automatic JSON config
    - Manual fit UI
    """
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
            if phase.get("fit_results") and "phase_time" in phase
        ]
        if fitted_phases:
            fig_summary = plot_phase_fit_with_ci(fitted_phases, operated_data, selected_operated_wells)
            st.plotly_chart(fig_summary, use_container_width=True)
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
        if st.button("Download Fit Configuration JSON"):
            st.download_button(
                label="💾 Download JSON",
                data=edited_json,
                file_name="fit_configuration.json",
                mime="application/json"
            )
    else:
        st.info("No fit results to summarize.")


def _handle_automatic_fits(uploaded_json_fit, operated_data, selected_wells):
    from utils.fitting import compute_confidence_intervals
    from utils.plotting import plot_fitted_curves, plot_confidence_intervals
    from scipy.optimize import curve_fit
    from scipy.stats import t as t_dist

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

            phase_data = operated_data[(operated_data["Time"]>=start) & (operated_data["Time"]<=end)][["Time"]+wells]
            phase_data = phase_data.dropna(subset=wells)
            if phase_data.empty:
                st.warning(f"No data points in interval {start} to {end}.")
                continue
            phase_data["Average"] = phase_data[wells].mean(axis=1)
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

            y_pred = model_func(time_vals, *popt)
            residuals = y_data - y_pred
            residual_variance = np.var(residuals, ddof=len(popt))
            dof = len(y_data) - len(popt)
            lower_bound_ci, upper_bound_ci = compute_confidence_intervals(time_vals, popt, pcov, 0.05, dof, residual_variance, model_func)
            perr = np.sqrt(np.diag(pcov))

            RSS = np.sum(residuals**2)
            AIC = 2*len(popt) + len(y_data)*np.log(RSS/len(y_data))
            t_statistic = popt / perr
            p_values = 2*(1 - t_dist.cdf(np.abs(t_statistic), df=dof))

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
                "Estimate": popt,
                "Std. Error": perr,
                "t-Statistic": t_statistic,
                "p-Value": p_values
            })
            st.dataframe(param_table)
    except Exception as e:
        st.error(f"Error processing JSON file: {e}")


def _display_existing_phases(operated_data, selected_operated_wells):
    from utils.plotting import plot_fitted_curves, plot_confidence_intervals
    from utils.models import MODEL_FUNCTIONS, MODEL_PARAMS, default_guesses
    from utils.fitting import compute_confidence_intervals
    from scipy.optimize import curve_fit
    from scipy.stats import t as t_dist

    for i, phase in enumerate(st.session_state.get("phases", [])):
        if "id" not in phase:
            phase["id"] = str(uuid.uuid4())
        phase_id = phase["id"]
        with st.expander(f"Manual Fit {i+1}"):
            try:
                t_interval = phase.get("time_interval", {})
                fit_start = st.text_input(f"Start Time for Fit {i+1}", value=str(t_interval.get("start", operated_data['Time'].min())), key=f"start_{i}")
                fit_end = st.text_input(f"End Time for Fit {i+1}", value=str(t_interval.get("end", operated_data['Time'].max())), key=f"end_{i}")
                phase["time_interval"] = {"start": float(fit_start), "end": float(fit_end)}

                # Delete Fit
                if st.button(f"Delete Fit {i+1}", key=f"delete_phase_{phase_id}"):
                    st.session_state["phases"].pop(i)
                    st.success(f"Deleted Fit {i+1}")
                    return

                phase_data = operated_data[(operated_data["Time"]>=float(fit_start)) & 
                                           (operated_data["Time"]<=float(fit_end))]
                if phase_data.empty:
                    st.warning("No data points in this interval.")
                    continue
                phase_data["Average"] = phase_data[selected_operated_wells].mean(axis=1)
                time_vals = phase_data["Time"].values
                y_data = phase_data["Average"].values

                phase["model"] = st.selectbox(
                    f"Select Model for Fit {i+1}",
                    [
                        "Exponential Growth",
                        "Logistic Growth",
                        "Baranyi Growth",
                        "Lag-Exponential-Saturation Growth",
                        "Gompertz Growth",  # Add Gompertz Growth here
                        "Custom Function",
                        "Automatic Fit"
                    ],
                    key=f"model_{i}"
                )
                if phase["model"] == "Custom Function":
                    phase["custom_model_expr"] = st.text_input(
                        f"Custom Model Expression for Fit {i+1}",
                        value=phase.get("custom_model_expr","X*exp(mu*t)"),
                        key=f"custom_model_expr_{i}"
                    )
                    phase["custom_params"] = st.text_input(
                        f"Parameters to Optimize (comma-separated) for Fit {i+1}",
                        value=", ".join(phase.get("custom_params",[])) or "X, mu",
                        key=f"custom_params_{i}"
                    ).split(",")
                    from utils.file_io import create_custom_model
                    model_func = create_custom_model(phase["custom_model_expr"], [p.strip() for p in phase["custom_params"] if p.strip()])
                    phase["parameters"] = [p.strip() for p in phase["custom_params"] if p.strip()]
                elif phase["model"] == "Automatic Fit":
                    st.info("Automatic Fit: evaluating candidate models.")
                    best_model, best_popt, best_pcov, best_aic, best_candidate = None,None,None,np.inf,None
                    from utils.models import MODEL_FUNCTIONS, MODEL_PARAMS, default_guesses

                    # Add Gompertz Growth to the list of candidate models
                    for candidate in [m for m in MODEL_FUNCTIONS.keys() if m not in ["Custom Function", "Automatic Fit"]]:
                        try:
                            cf_model_func = MODEL_FUNCTIONS[candidate]
                            guesses = default_guesses.get(candidate, [1.0]*len(MODEL_PARAMS.get(candidate,[])))
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
                        guess = st.number_input(f"Initial guess for {param}", value=1.0, step=0.01, format="%.5f", key=f"{param}_{i}_guess")
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
                        y_pred = model_func(time_vals, *popt)
                        residuals = y_data-y_pred
                        residual_variance = np.var(residuals, ddof=len(popt))
                        dof=len(y_data)-len(popt)
                        lower_bound_ci, upper_bound_ci = compute_confidence_intervals(time_vals, popt, pcov, 0.05, dof, residual_variance, model_func)
                        perr=np.sqrt(np.diag(pcov))
                        RSS=np.sum(residuals**2)
                        AIC=2*len(popt)+len(y_data)*np.log(RSS/len(y_data))
                        t_statistic=popt/perr
                        from scipy.stats import t as t_dist
                        p_values = 2*(1-t_dist.cdf(np.abs(t_statistic), df=dof))

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
                        plot_fitted_curves(phase_data, time_vals, y_data, y_pred, phase["model"])
                        plot_confidence_intervals(phase_data, lower_bound_ci, upper_bound_ci, y_pred, phase_data[selected_operated_wells].std(axis=1))

                        param_labels=[f"{p}" for p in phase.get("parameters",[])]
                        param_table=pd.DataFrame({
                            "Parameter":param_labels,
                            "Estimate":popt,
                            "Std. Error":perr,
                            "t-Statistic":t_statistic,
                            "p-Value":p_values
                        })
                        st.dataframe(param_table)
                    except Exception as e_fit:
                        st.error(f"Error fitting model for Fit {i+1}: {e_fit}")
            except Exception as e_manual:
                st.error(f"Error in manual fitting for Fit {i+1}: {e_manual}")
