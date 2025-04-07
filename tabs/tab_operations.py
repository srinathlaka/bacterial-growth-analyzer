import streamlit as st
from utils.operations import perform_group_operations
from utils.plotting import plot_selected_wells, plot_avg_sd_bg_subtracted

def display_tab_operations():
    """
    Tab 3: Operations on background-subtracted data (Add, Subtract, etc.).
    """
    st.subheader("Operations on Background-Subtracted Data")

    num_groups = st.session_state.get("num_groups", 1)
    groups_data = st.session_state.get("groups_data", {})

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
            operation = st.selectbox("Select operation", ["Add","Subtract","Multiply","Divide"])
            sample_wells_group1 = st.session_state.get("selected_sample_wells_by_group", {}).get(1, [])
            sample_wells_group2 = st.session_state.get("selected_sample_wells_by_group", {}).get(2, [])

            if sample_wells_group1 and sample_wells_group2:
                group1_data_samples = group1_data[["Time"] + sample_wells_group1]
                group2_data_samples = group2_data[["Time"] + sample_wells_group2]

                operated_data = perform_group_operations(group1_data_samples, group2_data_samples, operation)
                st.write(f"Result of {operation} operation between Group 1 and Group 2:")
                st.dataframe(operated_data)

                if "Average" not in operated_data.columns and operated_data.shape[1] > 1:
                    operated_data["Average"] = operated_data.iloc[:,1:].mean(axis=1)

                st.subheader("Plot Operated Data")
                plot_selected_wells(operated_data, operated_data.columns[1:])
                plot_avg_sd_bg_subtracted(operated_data, operated_data.columns[1:], group_num=999)  # pseudo group

                selected_operated_wells = operated_data.columns[1:].tolist()
                st.session_state["operated_data"] = operated_data
                st.session_state["selected_operated_wells"] = selected_operated_wells
        else:
            st.warning("Groups 1 or 2 do not exist. Please check Tab 2.")
    else:
        st.warning("No background-subtracted data is available. Please complete Tab 2 first.")
