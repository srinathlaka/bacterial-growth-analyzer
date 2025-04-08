# File: tabs/tab_background.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import os

from PIL import Image
from scipy.optimize import curve_fit

# Make sure these match your actual paths and function names
from utils.file_io import generate_labels, create_button_layout
from utils.background import perform_background_subtraction
from utils.plotting import (
    display_single_well_preview,     # Make sure this exists in your utils/plotting.py
    plot_selected_wells,
    plot_avg_sd_bg_subtracted
)
from utils.models import polynomial_growth, polynomial_func, MODEL_PARAMS
from utils.fitting import compute_confidence_intervals


def display_tab_background():
    """
    Tab 2: Background Subtraction.
    Allows optional JSON config or manual group specification.
    Lets user select blank & sample wells, subtract blanks, optionally fit blank wells, 
    and display background-corrected data.
    """
    st.header("Background Subtraction")

    # Ensure data from Tab 1 is loaded
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.info("Please upload a data file in Tab 1 first.")
        return
    df = st.session_state['df']

    # Initialize some session state variables
    if "selected_sample_wells_by_group" not in st.session_state:
        st.session_state["selected_sample_wells_by_group"] = {}
    if "groups_data" not in st.session_state:
        st.session_state["groups_data"] = {}

    # Use rows/columns from session (set in Tab 1), or default
    rows = st.session_state.get("rows", 8)
    columns = st.session_state.get("columns", 12)
    labels = generate_labels(rows, columns)

    # ----------------------------------------------------------------------
    # Section 1: JSON Upload (Optional)
    # ----------------------------------------------------------------------
    st.subheader("🔄 Upload JSON Configuration (Optional)")
    uploaded_json_config = st.file_uploader("Choose JSON Configuration File", type=["json"], key="tab2_json_upload")
    json_loaded = False
    groups_data = {}

    if uploaded_json_config is not None:
        try:
            config_data = json.load(uploaded_json_config)
            st.success("✅ JSON configuration loaded successfully.")
            json_loaded = True

            # Possibly override row/column layout with JSON
            layout = config_data.get("plate_layout", {})
            json_rows = layout.get("rows", rows)
            json_columns = layout.get("columns", columns)
            if json_rows != rows or json_columns != columns:
                st.warning(f"JSON layout ({json_rows}x{json_columns}) differs from current layout ({rows}x{columns}). Using JSON layout.")
                rows, columns = json_rows, json_columns
                st.session_state["rows"] = rows
                st.session_state["columns"] = columns
                labels = generate_labels(rows, columns)

            num_groups = config_data.get("num_groups", 1)
            st.session_state["num_groups"] = num_groups
            st.info(f"📊 Number of Groups (from JSON): {num_groups}")

            # For each group, parse info
            for group_num in range(1, num_groups + 1):
                group_str = str(group_num)
                group_info = config_data.get("groups", {}).get(group_str, {})
                blank_wells = group_info.get("blank_wells", [])
                sample_wells = group_info.get("sample_wells", [])
                fitting_info = group_info.get("fitting", {})

                st.session_state[f"group_{group_num}_blank_wells"] = blank_wells
                st.session_state[f"group_{group_num}_sample_wells"] = sample_wells
                st.session_state["selected_sample_wells_by_group"][group_num] = sample_wells

                st.write(f"**Group {group_num} (from JSON):**")
                st.write(f"Blank Wells: {', '.join(blank_wells) if blank_wells else 'None'}")
                st.write(f"Sample Wells: {', '.join(sample_wells) if sample_wells else 'None'}")
                st.write(f"Fitting Info: {fitting_info if fitting_info else 'Not specified'}")

                # Quick preview
                if blank_wells:
                    st.write("🔍 Preview of a Blank Well:")
                    # If one of the wells doesn't exist, handle it gracefully
                    for w in blank_wells:
                        if w not in df.columns:
                            st.warning(f"Column '{w}' not found in df.columns.")
                            continue
                        display_single_well_preview(df, w)
                if sample_wells:
                    st.write("🔍 Preview of a Sample Well:")
                    for w in sample_wells:
                        if w not in df.columns:
                            st.warning(f"Column '{w}' not found in df.columns.")
                            continue
                        display_single_well_preview(df, w)

                # Perform background subtraction
                groups_data = perform_background_subtraction(groups_data, df, group_num, blank_wells, sample_wells)

                # Optionally fit blank wells
                if blank_wells:
                    # Make sure blank wells exist in the DataFrame
                    missing_blank_cols = [bw for bw in blank_wells if bw not in df.columns]
                    if missing_blank_cols:
                        st.error(f"These blank wells don't exist in the data: {missing_blank_cols}")
                    else:
                        avg_blank = df[blank_wells].mean(axis=1)
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

                            # Compute confidence intervals
                            dof = len(df['Time']) - len(popt)
                            residual_variance = np.var(avg_blank - y_fit, ddof=len(popt))
                            lower_bound, upper_bound = compute_confidence_intervals(df['Time'], popt, pcov, 0.05, dof, residual_variance, model_func)

                            fig = _plot_blank_fit(df, avg_blank, y_fit, lower_bound, upper_bound, group_num)
                            st.plotly_chart(fig)
                        except Exception as e_fit:
                            st.error(f"Fitting failed for Group {group_num} blank wells: {e_fit}")

                # Now display background-subtracted results
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
    # Section 2: Manual Configuration (if no JSON file)
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
                selected_blank_wells = create_button_layout(rows, columns, labels, key_prefix=blank_wells_key, df=df)
                st.session_state[f"group_{group_num}_blank_wells"] = selected_blank_wells

                # Show a quick preview
                if selected_blank_wells:
                    st.markdown(f"**Preview of last selected blank well**: {selected_blank_wells[-1]}")
                    if selected_blank_wells[-1] not in df.columns:
                        st.warning(f"Column '{selected_blank_wells[-1]}' not found in DataFrame.")
                    else:
                        display_single_well_preview(df, selected_blank_wells[-1])

                if selected_blank_wells:
                    
                    _plot_average_blank(df, selected_blank_wells)
                    st.subheader(f"Fit Model to Blank Wells - Group {group_num}")
                    selected_model = st.selectbox(
                        f"Select Model for Blank Well Fitting - Group {group_num}",
                        ["Polynomial Growth", "Polynomial Function"],
                        key=f"model_{group_num}"
                    )
                    st.session_state[f"group_{group_num}_fitting"] = {"model": selected_model, "initial_guesses": [1.0, 1.0, 1.0]}

                    # Attempt to fit blank wells
                    avg_blank = None
                    missing_blank_cols = [bw for bw in selected_blank_wells if bw not in df.columns]
                    if missing_blank_cols:
                        st.error(f"These selected blank wells don't exist in the data: {missing_blank_cols}")
                    else:
                        # All columns exist
                        avg_blank = df[selected_blank_wells].mean(axis=1)
                    if avg_blank is not None:
                        if selected_model == "Polynomial Growth":
                            model_func = polynomial_growth
                        else:
                            model_func = polynomial_func
                        try:
                            popt, pcov = curve_fit(model_func, df['Time'], avg_blank)
                            y_pred = model_func(df['Time'], *popt)
                            st.success(f"Fitting for Group {group_num} blank wells successful.")

                            param_names = MODEL_PARAMS[selected_model]
                            param_df = pd.DataFrame({
                                "Parameter": param_names,
                                "Value": popt
                            })
                            st.write(f"### Fitted Parameters for Group {group_num} ({selected_model})")
                            st.dataframe(param_df)

                            dof = len(df['Time']) - len(popt)
                            residual_variance = np.var(avg_blank - y_pred, ddof=len(popt))
                            lower_bound, upper_bound = compute_confidence_intervals(df['Time'], popt, pcov, 0.05, dof, residual_variance, model_func)

                            fig2 = _plot_blank_fit(df, avg_blank, y_pred, lower_bound, upper_bound, group_num)
                            st.plotly_chart(fig2)
                        except Exception as e_fit:
                            st.error(f"Fitting failed for Group {group_num} blank wells: {e_fit}")

                st.write(f"🧪 Select sample wells for Group {group_num}")
                sample_wells_key = f"group_{group_num}_sample_wells"
                selected_sample_wells = create_button_layout(rows, columns, labels, key_prefix=sample_wells_key, df=df)
                st.session_state[f"group_{group_num}_sample_wells"] = selected_sample_wells
                st.session_state["selected_sample_wells_by_group"][group_num] = selected_sample_wells

                # Quick preview
                if selected_sample_wells:
                    st.markdown(f"**Preview of last selected sample well**: {selected_sample_wells[-1]}")
                    if selected_sample_wells[-1] not in df.columns:
                        st.warning(f"Column '{selected_sample_wells[-1]}' not found in DataFrame.")
                    else:
                        display_single_well_preview(df, selected_sample_wells[-1])

                if selected_sample_wells:
                    groups_data = perform_background_subtraction(groups_data, df, group_num, selected_blank_wells, selected_sample_wells)
                    st.session_state["groups_data"] = groups_data
                    group_df = groups_data.get(f"Group_{group_num}_bg_subtracted", None)
                    if group_df is not None:
                        st.write(f"### Background-Corrected Data for Group {group_num}")
                        st.dataframe(group_df)
                        plot_selected_wells(group_df, selected_sample_wells)
                        plot_avg_sd_bg_subtracted(group_df, selected_sample_wells, group_num)
                    else:
                        st.warning("No data available after background subtraction.")
                else:
                    st.warning(f"Group {group_num} sample wells have not been selected. Cannot perform background subtraction.")

    # ----------------------------------------------------------------------
    # Section 3: Editable JSON for Download
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


def _plot_blank_fit(df, avg_blank, y_fit, lower_bound, upper_bound, group_num):
    """
    Helper to plot blank well fit with confidence intervals.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=avg_blank, mode='lines', name='Observed Blank'))
    fig.add_trace(go.Scatter(x=df['Time'], y=y_fit, mode='lines', name='Fitted Curve', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Time'], y=upper_bound, mode='lines', name='Upper CI', line=dict(color='rgba(255,0,0,0.2)')))
    fig.add_trace(go.Scatter(x=df['Time'], y=lower_bound, mode='lines', name='Lower CI', line=dict(color='rgba(255,0,0,0.2)')))
    fig.update_layout(
        title=f"Fitting for Group {group_num} Blank Wells",
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    return fig


def _plot_average_blank(df, blank_wells):
    """
    Plot the average of the user-selected blank wells.
    Added a check to avoid KeyError if those wells don't exist.
    """
    # Check for missing wells first
    missing_cols = [bw for bw in blank_wells if bw not in df.columns]
    if missing_cols:
        st.error(f"Can't plot average blank because these columns don't exist: {missing_cols}")
        return

    import plotly.graph_objects as go
    import uuid

    avg_blank = df[blank_wells].mean(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=avg_blank, mode='lines', name='Avg Blank'))
    fig.update_layout(
        title='Average Blank Wells',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    st.plotly_chart(fig, key=f"plot_blank_{uuid.uuid4().hex}")
