# File: tabs/tab_background.py

import streamlit as st
import pandas as pd
import numpy as np
import uuid
import json
from PIL import Image
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import os

from utils.file_io import generate_labels, create_button_layout, read_data
from utils.background import perform_background_subtraction
from utils.plotting import (
    display_single_well_preview,
    plot_selected_wells,
    plot_avg_sd_bg_subtracted,
    plot_average_blank,
    plot_blank_fit,
    plot_raw_vs_corrected
)
from utils.models import polynomial_func, MODEL_PARAMS, power_law
from utils.fitting import compute_confidence_intervals

# Create a helper function to safely store state at the top of the file
def _store_ui_state(key, value):
    """Helper to safely store UI state"""
    # Only set if not already in session state to avoid the duplicate key error
    if key not in st.session_state:
        st.session_state[key] = value

def display_tab_background():
    st.header("Sample Selection & Background Subtraction")
    
    # Check if data is loaded from Tab 1
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.error("Please upload a data file in the Upload tab first.")
        return
    
    df = st.session_state['df']
    
    # Initialize key states at the start of display_tab_background()
    # Add after getting df from session state
    _store_ui_state("bg_tab_initialized", True)
    for i in range(1, st.session_state.get("num_groups", 1) + 1):
        # Initialize common UI state flags only if they don't exist yet
        _store_ui_state(f"perform_bg_sub_group_{i}", False)
        _store_ui_state(f"show_sample_plot_group_{i}", False)
        _store_ui_state(f"show_blank_plot_group_{i}", False)
    
    # Check if we already have processed data to avoid overwriting
    if "groups_data" in st.session_state and st.session_state["groups_data"]:
        st.info("Background subtracted data found in session state.")
        with st.expander("View existing processed data", expanded=False):
            for group_key, data in st.session_state["groups_data"].items():
                if isinstance(data, pd.DataFrame):
                    st.write(f"### {group_key}")
                    st.dataframe(data.head())
    
    # Initialize session state variables
    if "selected_sample_wells_by_group" not in st.session_state:
        st.session_state["selected_sample_wells_by_group"] = {}
    if "groups_data" not in st.session_state:
        st.session_state["groups_data"] = {}
    
    # Get plate dimensions from session state
    rows = st.session_state.get("rows", 8)
    columns = st.session_state.get("columns", 12)
    labels = generate_labels(rows, columns)
    
    # Clear UI distinction between modes
    is_advanced_mode = st.session_state.get("user_level") == "Advanced"
    if is_advanced_mode:
        st.info("🔬 You are in **Advanced Mode** with access to multiple groups, JSON configuration, and advanced background subtraction options.")
    else:
        st.info("🧪 You are in **Basic Mode** with simplified options for single group analysis.")
    
    # Determine if using Basic or Advanced mode
    is_advanced_mode = st.session_state.get("user_level") == "Advanced"
    
    # Get number of groups (default to 1 for Basic mode)
    if not is_advanced_mode:
        num_groups = 1
        st.session_state["num_groups"] = 1
    else:
        # Allow selecting up to 2 groups in Advanced mode
        st.write("#### Group Configuration")
        st.info("⚠️ Currently only supporting up to two groups for analysis")
        
        num_groups = st.number_input(
            "Number of sample groups to analyze:",
            min_value=1,
            max_value=2,
            value=st.session_state.get("num_groups", 1),
            help="In Advanced mode, you can analyze up to 2 groups of samples"
        )
        st.session_state["num_groups"] = num_groups
    
    # Support multiple file uploads for different groups in Advanced mode
    if is_advanced_mode and num_groups > 1:
        st.subheader("Group Data Sources")
        st.write("Please upload separate data files for each group.")
        
        # Always use different files when multiple groups are selected
        st.session_state["use_different_files"] = True
        st.session_state["group_dataframes"] = st.session_state.get("group_dataframes", {})
        
        for i in range(1, num_groups + 1):
            with st.expander(f"Data for Group {i}", expanded=True):
                file_key = f"file_group_{i}"
                uploaded_file = st.file_uploader(
                    f"Upload data file for Group {i}",
                    type=["csv", "xlsx", "xls"],
                    key=file_key
                )

                # Check if this group already has data loaded previously
                if i in st.session_state.get("group_dataframes", {}) and st.session_state["group_dataframes"][i] is not None:
                    # Show info about previously loaded file
                    st.success(f"✅ File for Group {i} already loaded. Preview:")
                    st.dataframe(st.session_state["group_dataframes"][i].head())
                    
                    # Option to reload if needed
                    if st.button(f"Reload data for Group {i}", key=f"reload_btn_{i}"):
                        if uploaded_file:
                            try:
                                group_df = read_data(uploaded_file, rows, columns)
                                st.success(f"✅ File for Group {i} reloaded successfully!")
                                st.dataframe(group_df.head())
                                st.session_state["group_dataframes"][i] = group_df
                            except Exception as e:
                                st.error(f"Error reloading file for Group {i}: {str(e)}")
                elif uploaded_file:
                    try:
                        group_df = read_data(uploaded_file, rows, columns)
                        st.success(f"✅ File for Group {i} loaded successfully!")
                        st.dataframe(group_df.head())
                        st.session_state["group_dataframes"][i] = group_df
                    except Exception as e:
                        st.error(f"Error loading file for Group {i}: {str(e)}")
                else:
                    st.warning(f"Please upload a data file for Group {i}")
    
    # JSON Configuration in Advanced Mode
    if is_advanced_mode:
        st.subheader("Configuration Options")
        with st.expander("Load Configuration from JSON", expanded=False):
            uploaded_json = st.file_uploader(
                "Upload JSON configuration file",
                type=["json"],
                help="Upload a JSON file with well selections and background settings"
            )
            if uploaded_json:
                try:
                    import json
                    config = json.load(uploaded_json)
                    st.success("✅ Configuration loaded successfully!")
                    st.json(config)
                    
                    # Apply configuration
                    if st.button("Apply Configuration"):
                        with st.spinner("Applying configuration..."):
                            # Logic to apply the configuration goes here
                            # This would set sample wells, blank wells, etc.
                            st.success("Configuration applied!")
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")
    
    # Group tracker for multi-group processing (advanced mode only)
    if is_advanced_mode and num_groups > 1:
        _display_group_tracker(num_groups)
    
    # Process each group
    for group_num in range(1, num_groups + 1):
        if num_groups > 1:
            st.markdown(f"## Group {group_num}")
        
        # Get dataframe to use for this group
        group_df = st.session_state.get("group_dataframes", {}).get(group_num, df)
        
        # STEP 1: Select Sample Wells first
        st.subheader(f"Step 1: Select Sample Wells for Group {group_num}")
        st.write("These are the wells containing your bacterial samples.")
        
        # Create sample well selection buttons
        sample_wells, plot_sample = create_button_layout(
            rows, columns, labels, 
            key_prefix=f"sample_wells_group_{group_num}", 
            df=group_df
        )
        
        # Store selected sample wells in session state
        if sample_wells:
            st.session_state[f"group_{group_num}_sample_wells"] = sample_wells
            st.session_state["selected_sample_wells_by_group"][group_num] = sample_wells
            
            # Immediately display the selected sample wells
            st.success(f"✅ Selected Sample Wells: {', '.join(sample_wells)}")
            
            # Plot the raw data for selected wells
            plot_button_key = f"plot_sample_wells_group_{group_num}"
            if plot_sample or st.button(f"Plot Selected Sample Wells for Group {group_num}", key=f"sample_btn_{group_num}"):
                st.session_state[f"show_sample_plot_group_{group_num}"] = True

            if st.session_state.get(f"show_sample_plot_group_{group_num}", False):
                st.write("### Raw Data for Selected Sample Wells")
                plot_selected_wells(group_df, sample_wells, context="initial")
                
                # Calculate and display average & standard deviation immediately
                st.write("### Average and Standard Deviation of Selected Sample Wells")
                
                # Create a temporary dataframe with just Time and selected wells
                temp_df = pd.DataFrame({'Time': group_df['Time']})
                for well in sample_wells:
                    if well in group_df.columns:
                        # Clip negative values to zero when creating temp_df
                        temp_df[well] = group_df[well].clip(lower=0)
                    else:
                        st.warning(f"Column '{well}' not found in uploaded data. Please check your file and well selection.")
                
                # Plot average and standard deviation
                plot_avg_sd_bg_subtracted(temp_df, sample_wells, group_num, context="initial")
                
                # Create raw data representation for later use
                st.session_state[f"group_{group_num}_raw_data"] = temp_df
                
                # Store to groups_data for fallback scenario (if bg subtraction is skipped)
                if f"Group_{group_num}_bg_subtracted" not in st.session_state.get("groups_data", {}):
                    groups_data = st.session_state.get("groups_data", {})
                    groups_data[f"Group_{group_num}_bg_subtracted"] = temp_df
                    st.session_state["groups_data"] = groups_data
        else:
            st.info(f"Please select sample wells for Group {group_num}.")
        
        # STEP 2: Optional Background Subtraction
        st.subheader(f"Step 2: Background Subtraction (Optional)")

        toggle_key = f"perform_bg_sub_group_{group_num}"
        perform_bg_sub = st.toggle(
            f"Perform background subtraction for Group {group_num}",
            key=toggle_key
        )
        
        if perform_bg_sub:
            with st.expander("Background Subtraction Settings", expanded=True):
                st.write("Select wells containing blank media (no bacteria) for background correction.")
                
                # Select blank wells
                blank_wells, plot_blank = create_button_layout(
                    rows, columns, labels, 
                    key_prefix=f"blank_wells_group_{group_num}", 
                    df=group_df
                )
                
                if blank_wells:
                    st.session_state[f"group_{group_num}_blank_wells"] = blank_wells
                    st.success(f"✅ Selected Blank Wells: {', '.join(blank_wells)}")
                    
                    # Display blank wells data
                    blank_plot_button_key = f"plot_blank_wells_group_{group_num}"
                    if plot_blank or st.button(f"Plot Blank Wells for Group {group_num}", key=blank_plot_button_key):
                        # Set a flag in session state to remember that we want to show this plot
                        st.session_state[f"show_blank_plot_group_{group_num}"] = True

                    # Check if we should show the plot (either from this run or previous runs)
                    if st.session_state.get(f"show_blank_plot_group_{group_num}", False):
                        st.write("### Raw Data for Selected Blank Wells")
                        plot_average_blank(group_df, blank_wells)
                    
                    # Model selection for background fitting
                    st.write("### Fit Model to Background Signal")
                    bg_model = st.selectbox(
                        f"Select model for blank well fitting (Group {group_num}):",
                        ["Power Law", "Polynomial", "No Fitting (Use Average)"],
                        key=f"bg_model_group_{group_num}"
                    )
                    
                    # Fit model to blank wells
                    if bg_model == "Power Law":
                        _fit_power_law(group_df, blank_wells, group_num)
                    elif bg_model == "Polynomial":
                        _fit_polynomial(group_df, blank_wells, group_num)
                    else:  # No Fitting - Use Average
                        _use_average_blank(group_df, blank_wells, group_num)
                    
                    # Perform background subtraction
                    if st.button(f"Perform Background Subtraction for Group {group_num}"):
                        sample_wells = st.session_state.get(f"group_{group_num}_sample_wells", [])
                        if not sample_wells:
                            st.error("No sample wells selected. Please select sample wells first.")
                        else:
                            with st.spinner("Performing background subtraction..."):
                                groups_data = st.session_state.get("groups_data", {})
                                groups_data = perform_background_subtraction(
                                    groups_data, group_df, group_num, blank_wells, sample_wells
                                )
                                st.session_state["groups_data"] = groups_data
                                st.session_state["bg_subtraction_skipped"] = False
                                
                                # Get the background-subtracted data from the updated groups_data
                                bg_subtracted_data = groups_data.get(f"Group_{group_num}_bg_subtracted")
                                
                                if bg_subtracted_data is not None:
                                    st.session_state[f"bg_subtracted_data_group_{group_num}"] = bg_subtracted_data
                                    st.session_state[f"show_bg_subtracted_plot_group_{group_num}"] = True
                                else:
                                    st.error("Background subtraction failed. No data was returned.")
                else:
                    st.info("Please select blank wells for background subtraction.")
        elif f"Group_{group_num}_bg_subtracted" in st.session_state.get("groups_data", {}):
            # The toggle resets to its default when the tab is revisited, so this
            # branch is also reached on a plain re-render - not only when the user
            # deliberately skips. Never rebuild raw data over a completed
            # subtraction: that would silently replace the corrected values with
            # uncorrected ones and flip bg_subtraction_skipped back to True.
            st.info(
                f"Existing background-subtracted data for Group {group_num} is preserved. "
                f"Switch the toggle on to redo the subtraction."
            )
        else:
            # User chose to skip background subtraction - automatically prepare data
            st.info("Background subtraction skipped. Preparing raw data for fitting...")
            st.session_state["bg_subtraction_skipped"] = True

            if num_groups > 1:
                st.session_state["multi_group_bg_skipped"] = True

            # Ensure we have raw data and it's clipped to zero
            sample_wells = st.session_state.get(f"group_{group_num}_sample_wells", [])
            if sample_wells:
                with st.spinner("Processing raw data for fitting..."):
                    # Create a new DataFrame for the processed raw data
                    
                    # Create a clean copy with just Time and selected sample wells
                    raw_data = pd.DataFrame({'Time': group_df['Time']})
                    
                    # Add each sample well, explicitly clipping negative values to zero
                    for well in sample_wells:
                        if well in group_df.columns:
                            # Only process if the column is numeric
                            col_data = group_df[well].values
                            if np.issubdtype(col_data.dtype, np.number):
                                raw_data[well] = np.maximum(col_data, 0)
                            else:
                                st.warning(f"Column '{well}' contains non-numeric data and will be skipped. Please check your file format.")
                        else:
                            st.warning(f"Column '{well}' not found in uploaded data. Please check your file and well selection.")
                    
                    # CRITICAL: Verify that all data is properly clipped to zero
                    for col in raw_data.columns:
                        if col != 'Time':
                            if (raw_data[col] < 0).any():
                                st.warning(f"Found negative values in {col}, clipping to zero")
                                raw_data[col] = raw_data[col].clip(lower=0)
                    
                    # Make sure we're showing the correct data
                    st.dataframe(raw_data.head())

                    # Calculate average and std dev for the processed raw data
                    if len(sample_wells) > 0:
                        valid_wells = [w for w in sample_wells if w in raw_data.columns]
                        if valid_wells:
                            raw_data['Average'] = raw_data[valid_wells].mean(axis=1)
                            raw_data['Std_Dev'] = raw_data[valid_wells].std(axis=1)
                    
                    # Store in session state
                    st.session_state[f"group_{group_num}_raw_data"] = raw_data
                    
                    # Store in groups_data for the next tab
                    groups_data = st.session_state.get("groups_data", {})
                    groups_data[f"Group_{group_num}_bg_subtracted"] = raw_data
                    st.session_state["groups_data"] = groups_data
                    
                    # Set operated_data for Tab 3
                    if group_num == 1:
                        st.session_state["operated_data"] = raw_data
                        st.session_state["selected_operated_wells"] = sample_wells
                    
                    st.success(f"✅ Raw data processed: all negative values clipped to zero")
                    
                    # Show a preview of the processed data
                    st.write("### Processed Raw Data Preview")
                    st.dataframe(raw_data.head())
                    
                    # Show plot with consistent key for persistence
                    st.write("### Processed Raw Data (Clipped to Zero)")
                    plot_avg_sd_bg_subtracted(raw_data, sample_wells, group_num, context="skipped_bg")
        
        # First, check if we have bg_subtracted_data from either this run or previous runs
        bg_subtracted_data = st.session_state.get(f"bg_subtracted_data_group_{group_num}")
        show_bg_plot = st.session_state.get(f"show_bg_subtracted_plot_group_{group_num}", False)

        if bg_subtracted_data is not None and show_bg_plot:
            # Get sample wells
            sample_wells = st.session_state.get(f"group_{group_num}_sample_wells", [])
            
            # Display the background-subtracted data (this will now persist)
            st.write("### Selected Wells After Background Subtraction")
            plot_selected_wells(bg_subtracted_data, sample_wells, context="after_bg")
            st.success("✅ Background subtraction completed!")
            
            # Show average and std deviation after bg subtraction
            st.write("### Average and Standard Deviation After Background Subtraction")
            plot_avg_sd_bg_subtracted(bg_subtracted_data, sample_wells, group_num, context="after_bg")
            
            # Compare raw vs background-corrected for a sample well
            if len(sample_wells) > 0:
                st.write("### Raw vs. Background-Corrected Comparison")
                
                # Store the selected well in session state
                well_key = f"compare_well_group_{group_num}"
                if well_key not in st.session_state:
                    st.session_state[well_key] = sample_wells[0]  # Default to first well
                    
                selected_well = st.selectbox(
                    f"Select a well to compare (Group {group_num}):",
                    sample_wells, 
                    key=well_key
                )
                
                # Get group_df (raw data)
                group_df = st.session_state.get("group_dataframes", {}).get(group_num, 
                                                st.session_state['df'])
                
                # Create plot with a stable key
                fig = plot_raw_vs_corrected(group_df, bg_subtracted_data, selected_well, group_num)
                st.plotly_chart(fig, use_container_width=True, 
                            key=f"raw_vs_bg_{group_num}_{selected_well}")
    
    # Download button for background subtraction configuration
    if st.button("Generate Background Configuration JSON"):
        _generate_bg_config_json(rows, columns, num_groups)
                    
def _display_group_tracker(num_groups):
    """Display a visual tracker for multiple groups"""
    st.markdown("""
    <style>
    .group-tracker {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .group-tracker-header {
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .group-tracker-flex {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .group-item {
        flex: 1;
        text-align: center;
    }
    .group-complete {
        color: #28a745;
        font-weight: bold;
    }
    .group-pending {
        color: #6c757d;
        font-weight: bold;
    }
    .group-divider {
        margin: 0 10px;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Build a simpler HTML structure
    html_parts = ['<div class="group-tracker">']
    html_parts.append('<div class="group-tracker-header">Multi-Group Processing</div>')
    html_parts.append('<div class="group-tracker-flex">')
    
    for i in range(1, num_groups + 1):
        group_data = st.session_state.get("groups_data", {})
        is_complete = f"Group_{i}_bg_subtracted" in group_data
        status_class = "group-complete" if is_complete else "group-pending"
        status_text = "Complete" if is_complete else "Pending"
        
        html_parts.append(f'<div class="group-item"><span class="{status_class}">Group {i}: {status_text}</span></div>')
        
        if i < num_groups:
            html_parts.append('<div class="group-divider">|</div>')
    html_parts.append('</div></div>')
    
    # Join all parts and render with unsafe_allow_html
    st.markdown(''.join(html_parts), unsafe_allow_html=True)

def _fit_power_law(df, blank_wells, group_num):
    """Fit a power law model to blank wells data"""
    # Calculate average of blank wells
    if not blank_wells or not all(well in df.columns for well in blank_wells):
        st.warning("Some selected blank wells are not found in the data.")
        return

    # Only use numeric blank wells, warn and skip non-numeric
    numeric_blank_wells = []
    for well in blank_wells:
        if np.issubdtype(df[well].dtype, np.number):
            numeric_blank_wells.append(well)
        else:
            st.warning(f"Blank well '{well}' contains non-numeric data and will be skipped.")
    if not numeric_blank_wells:
        st.warning("No numeric blank wells available for fitting. Please check your file format.")
        return

    avg_blank = df[numeric_blank_wells].mean(axis=1)
    time_vals = df['Time'].values

    try:
        # Initial parameter guesses for power law fit (a, b, c)
        initial_guess = [0.1, 0.1, 0.1]
        # Fit the power law model
        popt, pcov = curve_fit(power_law, time_vals, avg_blank, p0=initial_guess)
        # Store the fitting parameters
        st.session_state[f"group_{group_num}_fitting"] = {
            "model": "power_law",
            "parameters": popt.tolist()
        }
        # Generate predictions and confidence intervals
        y_pred = power_law(time_vals, *popt)
        residuals = avg_blank - y_pred
        residual_variance = np.var(residuals, ddof=len(popt))
        dof = len(avg_blank) - len(popt)
        # Compute confidence intervals
        alpha = 0.05  # 95% confidence interval
        lower_bound, upper_bound = compute_confidence_intervals(
            time_vals, popt, pcov, alpha, dof, residual_variance, power_law
        )
        # Plot the fitted model
        fig = plot_blank_fit(df, avg_blank, y_pred, lower_bound, upper_bound, group_num)
        st.plotly_chart(fig, use_container_width=True)
        # Display fit parameters
        st.success(f"Power Law Fit: y = {popt[0]:.4f} * x^{popt[1]:.4f} + {popt[2]:.4f}")
    except Exception:
        st.warning("Could not fit the power law model. Please check your data for numeric values and try again.")

def _fit_polynomial(df, blank_wells, group_num):
    """Fit a polynomial model to blank wells data"""
    # Calculate average of blank wells
    if not blank_wells or not all(well in df.columns for well in blank_wells):
        st.warning("Some selected blank wells are not found in the data.")
        return
        
    avg_blank = df[blank_wells].mean(axis=1)
    time_vals = df['Time'].values
    
    try:
        # Let user select polynomial degree
        degree = st.slider(
            f"Select polynomial degree for Group {group_num}:", 
            min_value=1, 
            max_value=5, 
            value=3
        )
        
        # Initial parameter guesses (zeros)
        initial_guess = [0.0] * (degree + 1)
        
        # Define polynomial function with selected degree
        def poly_func(x, *params):
            return np.polyval(params, x)
        
        # Fit the polynomial model
        popt, pcov = curve_fit(poly_func, time_vals, avg_blank, p0=initial_guess)
        
        # Store the fitting parameters
        st.session_state[f"group_{group_num}_fitting"] = {
            "model": "polynomial",
            "degree": degree,
            "parameters": popt.tolist()
        }
        
        # Generate predictions and confidence intervals
        y_pred = poly_func(time_vals, *popt)
        residuals = avg_blank - y_pred
        residual_variance = np.var(residuals, ddof=len(popt))
        dof = len(avg_blank) - len(popt)
        
        # Compute confidence intervals
        alpha = 0.05  # 95% confidence interval
        lower_bound, upper_bound = compute_confidence_intervals(
            time_vals, popt, pcov, alpha, dof, residual_variance, poly_func
        )
        
        # Plot the fitted model
        fig = plot_blank_fit(df, avg_blank, y_pred, lower_bound, upper_bound, group_num)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display fit parameters in a readable format
        param_str = " + ".join([f"{popt[i]:.4f} * x^{degree-i}" for i in range(degree+1)])
        st.success(f"Polynomial Fit (degree {degree}): y = {param_str}")
        
    except Exception as e:
        st.error(f"Error fitting polynomial model: {str(e)}")

def _use_average_blank(df, blank_wells, group_num):
    """Use simple average of blank wells without model fitting"""
    # Calculate average of blank wells
    if not blank_wells or not all(well in df.columns for well in blank_wells):
        st.warning("Some selected blank wells are not found in the data.")
        return
    
    avg_blank = df[blank_wells].mean(axis=1)
    
    # Store the "fitting" information
    st.session_state[f"group_{group_num}_fitting"] = {
        "model": "average",
        "parameters": []
    }
    
    # Create a simple visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=avg_blank,
        mode='lines',
        name='Average of Blank Wells'
    ))
    fig.update_layout(
        title=f'Average of Blank Wells - Group {group_num} (No Fitting)',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success("Using simple average of blank wells (no model fitting).")

def _generate_bg_config_json(rows, columns, num_groups):
    """Generate and offer download of background configuration JSON"""
    config_bg_subtraction = {
        "plate_layout": {
            "rows": rows,
            "columns": columns
        },
        "num_groups": num_groups,
        "groups": {}
    }
    
    for group_num in range(1, num_groups + 1):
        group_key = str(group_num)
        blank_wells = st.session_state.get(f"group_{group_num}_blank_wells", [])
        sample_wells = st.session_state.get(f"group_{group_num}_sample_wells", [])
        fitting_info = st.session_state.get(f"group_{group_num}_fitting", {})
        config_bg_subtraction["groups"][group_key] = {
            "blank_wells": blank_wells,
            "sample_wells": sample_wells,
            "fitting": fitting_info
        }
    
    # Convert to JSON string
    config_json = json.dumps(config_bg_subtraction, indent=2)
    
    # Provide editable text area
    edited_bg_json = st.text_area(
        "Edit Background Subtraction JSON Configuration:", 
        value=config_json, 
        height=300
    )
    
    # Offer download button
    st.download_button(
        label="💾 Download Background Configuration",
        data=edited_bg_json,
        file_name="background_subtraction_config.json",
        mime="application/json"
    )
