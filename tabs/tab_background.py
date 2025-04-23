# File: tabs/tab_background.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import os
import uuid

from PIL import Image
from scipy.optimize import curve_fit

# Make sure these match your actual paths and function names
from utils.file_io import generate_labels, create_button_layout, read_data
from utils.background import perform_background_subtraction
from utils.plotting import (
    display_single_well_preview,     # Make sure this exists in your utils/plotting.py
    plot_selected_wells,
    plot_avg_sd_bg_subtracted,
    plot_average_blank,
    plot_blank_fit,
    plot_raw_vs_corrected
)
from utils.models import polynomial_func, MODEL_PARAMS, power_law
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
                        if model_used == "Power Law":
                            model_func = power_law
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

                            fig = plot_blank_fit(df, avg_blank, y_fit, lower_bound, upper_bound, group_num)
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

        # Only show group organization options if more than 1 group
        if num_groups > 1:
            st.subheader("🔍 Group Data Source Configuration")
            group_data_mode = st.radio(
                "How are your groups organized?",
                ["All groups use the same dataset", 
                 "Assign specific wells to each group",
                 "Upload separate files for each group"],
                help="Choose how your group data is organized"
            )
        else:
            # Default to simple mode for single group
            group_data_mode = "All groups use the same dataset"

        # Initialize group_dataframes if not exists
        if "group_dataframes" not in st.session_state:
            st.session_state["group_dataframes"] = {}

        # Set default for group 1 using main data
        if 'df' in st.session_state and st.session_state['df'] is not None:
            if 1 not in st.session_state["group_dataframes"]:
                st.session_state["group_dataframes"][1] = df

        # Handle specific wells assignment (this is already in your code)
        if group_data_mode == "Assign specific wells to each group":
            if "group_well_assignments" not in st.session_state:
                st.session_state["group_well_assignments"] = {i: [] for i in range(1, num_groups+1)}
            
            st.write("### Assign Wells to Groups")
            st.write("Select which wells belong to each group. Wells can be assigned to multiple groups if needed.")
            
            for group_num in range(1, num_groups + 1):
                with st.expander(f"Group {group_num} Well Assignment", expanded=False):
                    st.write(f"🔬 Select wells that belong to Group {group_num}")
                    group_wells_key = f"group_{group_num}_assigned_wells"
                    assigned_wells, _ = create_button_layout(rows, columns, labels, key_prefix=group_wells_key, df=df)
                    st.session_state["group_well_assignments"][group_num] = assigned_wells
                    
                    if assigned_wells:
                        st.success(f"✅ {len(assigned_wells)} wells assigned to Group {group_num}")
                    else:
                        st.warning(f"⚠️ No wells assigned to Group {group_num}")

        # Add new section for separate file uploads
        elif group_data_mode == "Upload separate files for each group":
            st.write("### Upload Files for Each Group")
            
            for group_num in range(1, num_groups + 1):
                with st.expander(f"Group {group_num} Data File", expanded=True):
                    uploaded_file = st.file_uploader(
                        f"Upload data file for Group {group_num}", 
                        type=["xlsx", "csv"], 
                        key=f"group_{group_num}_file"
                    )
                    
                    if uploaded_file is not None:
                        with st.spinner(f"Reading data for Group {group_num}..."):
                            group_df = read_data(uploaded_file, rows, columns)
                            if group_df is not None:
                                st.session_state["group_dataframes"][group_num] = group_df
                                st.success(f"✅ Data for Group {group_num} loaded successfully")
                                view_raw_data = st.checkbox(f"View Raw Data for Group {group_num}", key=f"view_raw_{group_num}")
                                if view_raw_data:
                                    st.dataframe(group_df)
                    else:
                        st.info(f"Please upload a file for Group {group_num}")

        for group_num in range(1, num_groups + 1):
            st.subheader(f"Group {group_num} Well Selection for Background Correction")
            with st.expander(f"Group {group_num} Blank and Sample Well Selection", expanded=True):
                # Get the correct DataFrame for this group based on data mode
                if group_data_mode == "Upload separate files for each group":
                    if group_num in st.session_state["group_dataframes"]:
                        group_df = st.session_state["group_dataframes"][group_num]
                        df_label = f"Uploaded data for Group {group_num}"
                    else:
                        st.error(f"No data file uploaded for Group {group_num}")
                        continue
                elif group_data_mode == "Assign specific wells to each group":
                    # Determine which wells are available for this group
                    available_wells = st.session_state["group_well_assignments"].get(group_num, [])
                    if not available_wells:
                        st.error(f"No wells assigned to Group {group_num}. Please assign wells above.")
                        continue
                    
                    # Create filtered dataframe with only the assigned wells
                    group_df = df[["Time"] + available_wells]
                    df_label = "Filtered data for this group"
                else:
                    # Use the full dataset (original behavior)
                    group_df = df
                    df_label = "Full dataset"
                    
                # Show what data is being used for this group
                st.write(f"**Using {df_label}** ({len(group_df.columns)-1} wells available)")
                
                # Now use group_df instead of df for this group's operations
                st.write(f"🧫 Select blank wells for Group {group_num}")
                blank_wells_key = f"group_{group_num}_blank_wells"
                selected_blank_wells, plot_blank = create_button_layout(rows, columns, labels, 
                                                                        key_prefix=blank_wells_key, 
                                                                        df=group_df)  # Use group_df here

                st.session_state[f"group_{group_num}_blank_wells"] = selected_blank_wells

                # Show a quick preview
                if selected_blank_wells:
                    if plot_blank:  # Store in session state when Plot Selected is clicked
                        st.session_state[f"group_{group_num}_plot_blank_shown"] = True
                        st.session_state[f"group_{group_num}_selected_blank_wells"] = selected_blank_wells
                        
                    # Display plots if Plot Selected was just clicked OR we've shown plots before
                    if plot_blank or st.session_state.get(f"group_{group_num}_plot_blank_shown", False):
                        blanks_to_show = st.session_state.get(f"group_{group_num}_selected_blank_wells", selected_blank_wells)
                        if blanks_to_show:
                            # First, just plot the selected wells like in Tab 1
                            st.subheader(f"Selected Blank Wells - Group {group_num}")
                            plot_selected_wells(group_df, blanks_to_show)
                            plot_average_blank(group_df, blanks_to_show)
                            
                            # Add a separate button for fitting
                            # First, show the model selection before the fit button
                            st.subheader(f"Fit Model to Blank Wells - Group {group_num}")
                            selected_model = st.selectbox(
                                f"Select Model for Blank Well Fitting - Group {group_num}",
                                ["Power Law", "Polynomial Function"],  # These are your available models
                                key=f"model_{group_num}"
                            )
                            st.session_state[f"group_{group_num}_fitting"] = {"model": selected_model, "initial_guesses": [1.0, 1.0, 1.0]}

                            # Then show the fit button (which now includes the selected model name)
                            if st.button(f"Fit Curve Using {selected_model} - Group {group_num}", key=f"fit_blank_{group_num}"):
                                # Attempt to fit blank wells
                                avg_blank = None
                                missing_blank_cols = [bw for bw in selected_blank_wells if bw not in group_df.columns]
                                if missing_blank_cols:
                                    st.error(f"These selected blank wells don't exist in the data: {missing_blank_cols}")
                                else:
                                    # All columns exist
                                    avg_blank = group_df[selected_blank_wells].mean(axis=1)
                                if avg_blank is not None:
                                    if selected_model == "Power Law":
                                        model_func = power_law
                                    else:
                                        model_func = polynomial_func
                                    try:
                                        popt, pcov = curve_fit(model_func, group_df['Time'], avg_blank)
                                        y_pred = model_func(group_df['Time'], *popt)
                                        
                                        # Calculate standard errors
                                        perr = np.sqrt(np.diag(pcov))
                                        
                                        # Store results in session state to persist between reruns
                                        fit_key = f"group_{group_num}_blank_fit_results"
                                        st.session_state[fit_key] = {
                                            "popt": popt,
                                            "pcov": pcov,
                                            "perr": perr,  # Add standard errors
                                            "y_pred": y_pred,
                                            "avg_blank": avg_blank,
                                            "param_names": MODEL_PARAMS[selected_model],
                                            "model": selected_model
                                        }
                                        
                                        st.success(f"Fitting for Group {group_num} blank wells successful.")

                                        param_names = MODEL_PARAMS[selected_model]
                                        param_df = pd.DataFrame({
                                            "Parameter": param_names,
                                            "Value": popt,
                                            "Std. Error": perr  # Add standard errors to the dataframe
                                        })
                                        st.write(f"### Fitted Parameters for Group {group_num} ({selected_model})")
                                        st.dataframe(param_df)

                                        dof = len(group_df['Time']) - len(popt)
                                        residual_variance = np.var(avg_blank - y_pred, ddof=len(popt))
                                        lower_bound, upper_bound = compute_confidence_intervals(group_df['Time'], popt, pcov, 0.05, dof, residual_variance, model_func)
                                        
                                        # Store CI results too
                                        st.session_state[f"group_{group_num}_blank_fit_ci"] = {
                                            "lower_bound": lower_bound,
                                            "upper_bound": upper_bound
                                        }

                                        fig2 = plot_blank_fit(group_df, avg_blank, y_pred, lower_bound, upper_bound, group_num)
                                        st.plotly_chart(fig2, key=f"blank_fit_plot_{group_num}_{uuid.uuid4().hex}")
                                    except Exception as e_fit:
                                        st.error(f"Fitting failed for Group {group_num} blank wells: {e_fit}")
                        else:
                            st.info("Click 'Plot Selected' to view and fit the blank wells.")

                        # Display previous fitting results if they exist
                        fit_key = f"group_{group_num}_blank_fit_results"
                        ci_key = f"group_{group_num}_blank_fit_ci"

                        # Check if we have previous fitting results to display
                        if fit_key in st.session_state and ci_key in st.session_state:
                            # Display results from previous fit
                            fit_results = st.session_state[fit_key]
                            ci_results = st.session_state[ci_key]
                            
                            # Extract the values we need
                            model = fit_results["model"]
                            popt = fit_results["popt"]
                            perr = fit_results.get("perr", np.zeros_like(popt))  # Get perr if available, else zeros
                            y_pred = fit_results["y_pred"]
                            avg_blank = fit_results["avg_blank"]
                            param_names = fit_results["param_names"]
                            lower_bound = ci_results["lower_bound"]
                            upper_bound = ci_results["upper_bound"]
                            
                            # Display the fitted parameters
                            st.write(f"### Previously Fitted Parameters for Group {group_num} ({model})")
                            param_df = pd.DataFrame({
                                "Parameter": param_names,
                                "Value": popt,
                                "Std. Error": perr  # Add standard errors to the previous results too
                            })
                            st.dataframe(param_df)
                            
                            # Display the plot using the stored results
                            fig2 = plot_blank_fit(group_df, avg_blank, y_pred, lower_bound, upper_bound, group_num)
                            st.plotly_chart(fig2, key=f"prev_blank_fit_plot_{group_num}_{uuid.uuid4().hex}")

                st.write(f"🧪 Select sample wells for Group {group_num}")
                sample_wells_key = f"group_{group_num}_sample_wells"
                selected_sample_wells, plot_sample = create_button_layout(rows, columns, labels, 
                                                                          key_prefix=sample_wells_key, 
                                                                          df=group_df)  # Use group_df here
                st.session_state[f"group_{group_num}_sample_wells"] = selected_sample_wells
                st.session_state["selected_sample_wells_by_group"][group_num] = selected_sample_wells

                # Quick preview
                if selected_sample_wells and isinstance(selected_sample_wells, list) and len(selected_sample_wells) > 0:
                    last_well = selected_sample_wells[-1]
                    st.markdown(f"**Preview of last selected sample well**: {last_well}")
                    if last_well not in group_df.columns:
                        st.warning(f"Column '{last_well}' not found in DataFrame.")
                    else:
                        display_single_well_preview(group_df, last_well)

                # Only process when Plot Selected is clicked for samples
                if selected_sample_wells:
                    if plot_sample:  # Just clicked Plot Selected
                        st.session_state[f"group_{group_num}_plot_sample_shown"] = True
                        st.session_state[f"group_{group_num}_selected_sample_wells"] = selected_sample_wells
                        
                        # Perform background subtraction
                        groups_data = perform_background_subtraction(groups_data, group_df, group_num, selected_blank_wells, selected_sample_wells)
                        st.session_state["groups_data"] = groups_data
                        
                        # Store the background-subtracted dataframe
                        group_bg_subtracted = groups_data.get(f"Group_{group_num}_bg_subtracted", None)
                        if group_bg_subtracted is not None:
                            st.session_state[f"group_{group_num}_bg_subtracted"] = group_bg_subtracted
                    
                    # Display if we've plotted before or just clicked Plot Selected
                    if plot_sample or st.session_state.get(f"group_{group_num}_plot_sample_shown", False):
                        samples_to_show = st.session_state.get(f"group_{group_num}_selected_sample_wells", selected_sample_wells)
                        if samples_to_show:
                            # Show original sample data first (raw data)
                            st.subheader(f"Selected Sample Wells (Raw) - Group {group_num}")
                            plot_selected_wells(group_df, samples_to_show)
                            
                            # Then show background-subtracted data if it exists
                            bg_subtracted_df = st.session_state.get(f"group_{group_num}_bg_subtracted")
                            if bg_subtracted_df is not None or f"Group_{group_num}_bg_subtracted" in groups_data:
                                if bg_subtracted_df is None:
                                    bg_subtracted_df = groups_data.get(f"Group_{group_num}_bg_subtracted") 
                                    
                                st.write(f"### Background-Corrected Data for Group {group_num}")
                                st.dataframe(bg_subtracted_df)
                                plot_selected_wells(bg_subtracted_df, samples_to_show)
                                plot_avg_sd_bg_subtracted(bg_subtracted_df, samples_to_show, group_num)
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
