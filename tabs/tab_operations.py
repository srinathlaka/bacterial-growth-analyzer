import numpy as np
import pandas as pd
import streamlit as st
from utils.operations import perform_group_operations
from utils.plotting import plot_selected_wells, plot_avg_sd_bg_subtracted

def display_tab_operations():
    st.header("Data Operations")
    
    # Add defensive data recovery
    if "operated_data" in st.session_state and (
        "groups_data" not in st.session_state or 
        not st.session_state["groups_data"] or
        "Group_1_bg_subtracted" not in st.session_state.get("groups_data", {})
    ):
        # Recover missing data
        st.warning("⚠️ Restoring groups_data from operated_data")
        groups_data = {}
        operated_data = st.session_state["operated_data"].copy()
        groups_data["Group_1_bg_subtracted"] = operated_data
        st.session_state["groups_data"] = groups_data
    
    # Debug information (hidden unless debug mode is enabled in the sidebar)
    if st.session_state.get("debug_mode", False):
        with st.expander("Debug Information"):
            st.write(f"groups_data keys: {list(st.session_state.get('groups_data', {}).keys())}")
            st.write(f"bg_subtraction_skipped: {st.session_state.get('bg_subtraction_skipped', False)}")
            st.write(f"multi_group_bg_skipped: {st.session_state.get('multi_group_bg_skipped', False)}")
            st.write(f"operated_data exists: {'operated_data' in st.session_state}")
            for i in range(1, 5):  # Check for up to 4 groups
                st.write(f"group_{i}_raw_data exists: {f'group_{i}_raw_data' in st.session_state}")


    # Cross-copy data from operated_data to groups_data if needed
    if "operated_data" in st.session_state and not st.session_state.get("groups_data"):
        # Set up groups_data
        groups_data = {}
        operated_data = st.session_state["operated_data"].copy()
        groups_data["Group_1_bg_subtracted"] = operated_data
        st.session_state["groups_data"] = groups_data
        st.info("Automatically copied operated_data to groups_data")
    
    # Check if background subtraction was skipped with multiple groups
    if st.session_state.get("multi_group_bg_skipped", False):
        st.warning("""
        ⚠️ **Background subtraction was skipped for multiple groups**
        
        You're working with raw data without background correction.
        You can now perform operations between groups before proceeding to fitting.
        """)
        
        # Change the subheader to reflect we're working with raw data
        st.subheader("Operations on Raw Data (No Background Subtraction)")
        
        # Get raw data from session state
        group1_raw = st.session_state.get("group_1_raw_data")
        group2_raw = st.session_state.get("group_2_raw_data")
        
        if group1_raw is not None and group2_raw is not None:
            # Replace the normal groups_data with raw data
            groups_data = {
                "Group_1_bg_subtracted": group1_raw,
                "Group_2_bg_subtracted": group2_raw
            }
            st.session_state["groups_data"] = groups_data
            
            # Continue with the rest of operations tab...
            
        else:
            st.error("Could not find raw data for all groups. Please go back to the Background tab.")
            return
    else:
        st.subheader("Operations on Background-Subtracted Data")

    # CRITICAL: Ensure all data passed to fitting has no negative values
    # This ensures even if raw data somehow gets through, it will be clipped
    if "groups_data" in st.session_state:
        groups_data = st.session_state["groups_data"]
        for key in groups_data:
            df = groups_data[key]
            if isinstance(df, pd.DataFrame):
                # Skip the Time column when clipping
                for col in df.columns:
                    if col != 'Time':
                        # Clip any negative values to zero
                        if (df[col] < 0).any():
                            df[col] = df[col].clip(lower=0)
                groups_data[key] = df
        # Update session state with cleaned data
        st.session_state["groups_data"] = groups_data
    
    # Also ensure operated_data has no negative values
    if "operated_data" in st.session_state:
        operated_data = st.session_state["operated_data"]
        for col in operated_data.columns:
            if col != 'Time':
                operated_data[col] = operated_data[col].clip(lower=0)
        st.session_state["operated_data"] = operated_data

    # Get data - either background subtracted or raw
    num_groups = st.session_state.get("num_groups", 1)
    groups_data = st.session_state.get("groups_data", {})

    # Show error if no data is available
    if not groups_data:
        st.error("No background-subtracted data is available. Please complete Tab 2 first.")
        return

    if num_groups == 1 and groups_data:
        st.info("Only one group is selected. Using that group’s background-subtracted data.")
        group1_data = groups_data.get("Group_1_bg_subtracted")
        sample_wells_group1 = st.session_state.get("selected_sample_wells_by_group", {}).get(1, [])

        if group1_data is not None and sample_wells_group1:
            st.write("Background-subtracted data for Group 1:")
            st.dataframe(group1_data)

            st.subheader("Plot Background-Subtracted Data for Group 1")
            plot_selected_wells(group1_data, sample_wells_group1)
            plot_avg_sd_bg_subtracted(group1_data, sample_wells_group1, group_num=1)

            operated_data = group1_data
            selected_operated_wells = sample_wells_group1
            st.session_state["operated_data"] = operated_data
            st.session_state["selected_operated_wells"] = selected_operated_wells

        else:
            st.warning("No valid group 1 data or sample wells found. Please check Tab 2.")
    elif num_groups > 1 and groups_data:
        group1_data = groups_data.get("Group_1_bg_subtracted")
        group2_data = groups_data.get("Group_2_bg_subtracted")

        if group1_data is not None and group2_data is not None:
            group_order = st.radio(
                "Select group order for operation",
                ["Group 1 [operation] Group 2", "Group 2 [operation] Group 1"],
                key="group_order"
            )
            
            operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])
            sample_wells_group1 = st.session_state.get("selected_sample_wells_by_group", {}).get(1, [])
            sample_wells_group2 = st.session_state.get("selected_sample_wells_by_group", {}).get(2, [])

            if sample_wells_group1 and sample_wells_group2:
                group1_data_samples = group1_data[["Time"] + sample_wells_group1]
                group2_data_samples = group2_data[["Time"] + sample_wells_group2]

                if group_order == "Group 1 [operation] Group 2":
                    operated_data = perform_group_operations(group1_data_samples, group2_data_samples, operation)
                    operation_description = f"Group 1 {operation} Group 2"
                else:
                    operated_data = perform_group_operations(group2_data_samples, group1_data_samples, operation)
                    operation_description = f"Group 2 {operation} Group 1"
                    
                st.write(f"Result of {operation_description}:")
                st.dataframe(operated_data)

                if "Average" not in operated_data.columns and operated_data.shape[1] > 1:
                    operated_data["Average"] = operated_data.iloc[:,1:].mean(axis=1)

                st.subheader("Plot Operated Data")
                # Filter out the Time column and any other non-well columns
                well_columns = [col for col in operated_data.columns if col != 'Time' and col != 'Average' and col != 'Std_Dev']
                plot_selected_wells(operated_data, well_columns, context="operations")
                plot_avg_sd_bg_subtracted(operated_data, operated_data.columns[1:], group_num=999)  # pseudo group

                selected_operated_wells = operated_data.columns[1:].tolist()
                st.session_state["operated_data"] = operated_data
                st.session_state["selected_operated_wells"] = selected_operated_wells
        else:
            st.warning("Groups 1 or 2 do not exist. Please check Tab 2.")
    else:
        st.warning("No background-subtracted data is available. Please complete Tab 2 first.")
