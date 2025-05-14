import streamlit as st
import os
from PIL import Image
import json
import pandas as pd

from utils.file_io import (
    read_data,
    create_example_data,
    generate_labels,
    create_button_layout
)
from utils.plotting import plot_selected_wells

def display_tab_upload():
    """
    Tab 1: Upload and Inspect Data. Lets user pick layout, upload file, 
    see raw data, and manually select wells for preview.
    """
    # Initialize df to None at the start to avoid UnboundLocalError
    df = None
    
    st.header("📁 Raw Data Plotting")

    # Show different content based on user mode
    if st.session_state.get("user_level") == "Basic":
        st.info("""
        ### Basic Mode Tutorial
        
        Welcome to the Bacterial Growth Analyzer! This simplified mode will help you:
        1. **Upload** your plate reader data
        2. **Subtract** background readings
        3. **Fit** growth models to your data
        
        Need help getting started? Check the information below!
        """)
    else:
        st.info("""
        ### Advanced Mode Features
        
        All features are unlocked in Advanced Mode:
        - Multi-group data analysis
        - Custom mathematical operations
        - ODE-based analysis
        - Custom function fitting
        - Multiple data sources
        """)

    # Layout selection
    rows, columns = _select_layout()
    st.session_state["rows"] = rows
    st.session_state["columns"] = columns
    labels = generate_labels(rows, columns)

    # Limit groups in Basic mode
    if st.session_state.get("user_level") == "Basic" and "num_groups" not in st.session_state:
        # Basic mode - force to single group
        st.session_state["num_groups"] = 1
    else:
        # Allow multiple groups in Advanced mode
        if st.session_state.get("user_level") == "Advanced" and "num_groups" not in st.session_state:
            num_groups = st.number_input("Number of groups/experiments", min_value=1, max_value=4, value=1)
            st.session_state["num_groups"] = num_groups

    # Display example and images if exist
    _display_example_and_images()

    st.subheader("📂 Upload Your Data File (CSV or XLSX)")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx","csv"], key="data_file")

    # Add this right after the file uploader
    if uploaded_file is None and "df" in st.session_state:
        # Show a notice that data is already loaded
        st.success(f"""
        ✅ **Data already loaded!** 
        
        Your previous file is still available in memory even though the uploader shows empty.
        You can continue to work with your data in other tabs.
        """)
        
        # Option to clear data if desired
        if st.button("Clear loaded data", key="clear_data"):
            # Remove relevant keys
            keys_to_remove = ["df", "upload_tab_selected_wells", "groups_data", "num_groups"]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Use the df that's already in session state
        df = st.session_state["df"]
        
        # Optional: show a preview of the data
        with st.expander("📊 View Raw Data"):
            st.dataframe(df)
            
    elif uploaded_file is not None:
        with st.spinner("Reading and processing the data..."):
            # Add a warning if previous data exists
            if ("groups_data" in st.session_state and st.session_state["groups_data"]) or \
               ("operated_data" in st.session_state and isinstance(st.session_state["operated_data"], pd.DataFrame) and not st.session_state["operated_data"].empty):
                st.warning("""
                ⚠️ **Warning:** Previous analysis data exists in memory!
                
                For accurate analysis with your new file, it's recommended to reset the app.
                """)
                
                # Add a reset app button
                if st.button("🔄 Reset Entire App", key="reset_entire_app"):
                    # List of all keys that should be preserved
                    keys_to_preserve = ["user_level", "current_tab"]
                    
                    # Store the values we want to keep
                    preserved_values = {k: st.session_state[k] for k in keys_to_preserve if k in st.session_state}
                    
                    # Clear the entire session state
                    for key in list(st.session_state.keys()):
                        if key not in keys_to_preserve:
                            del st.session_state[key]
                    
                    # Restore preserved values
                    for k, v in preserved_values.items():
                        st.session_state[k] = v
                    
                    st.success("✅ App has been reset! All previous analysis data has been cleared.")
                    st.rerun()
            
            # Continue with reading the new file
            df = read_data(uploaded_file, rows, columns)
            if df is not None:
                st.session_state["df"] = df
                with st.expander("📊 View Raw Data"):
                    st.dataframe(df)
    else:
        st.info("Please upload a file to continue.")

    if df is not None:
        st.subheader("🔬 Select Wells to Plot")
        selected_wells, plot_selected = create_button_layout(rows, columns, labels, key_prefix="tab1_wells", df=df)
        
        # Store selected wells in session state
        if selected_wells:
            st.session_state["upload_tab_selected_wells"] = selected_wells
            st.success(f"✅ Selected Wells: {', '.join(selected_wells)}")

            # If Plot button was just clicked OR we have previous plot data to show
            if plot_selected:
                # Plot the data and store the fact that we plotted
                plot_selected_wells(df, selected_wells)
                st.session_state["upload_tab_has_plot"] = True
            elif st.session_state.get("upload_tab_has_plot", False):
                # Re-plot using previous selection if we had plotted before
                stored_wells = st.session_state.get("upload_tab_selected_wells", [])
                if stored_wells:
                    plot_selected_wells(df, stored_wells)
        else:
            # Clear the plot state if no wells are selected
            st.session_state["upload_tab_has_plot"] = False
            st.info("🛑 Please select some wells.")


def _select_layout():
    """
    Simplified layout selection with a single dropdown menu.
    Custom option triggers additional inputs.
    """
    well_format = st.selectbox(
        "Select well format",
        [
            "96 well rows 8 column 12",
            "24 well rows 4 column 6",
            "84 well rows 7 column 12",
            "1536 well rows 32 column 48",
            "Custom"
        ]
    )
    
    if well_format == "Custom":
        custom_rows = st.number_input("Enter number of rows", min_value=1, value=8, step=1)
        custom_columns = st.number_input("Enter number of columns", min_value=1, value=12, step=1)
        return int(custom_rows), int(custom_columns)
    elif well_format == "24 well rows 4 column 6":
        return 4, 6
    elif well_format == "96 well rows 8 column 12":
        return 8, 12
    elif well_format == "84 well rows 7 column 12":
        return 7, 12
    elif well_format == "1536 well rows 32 column 48":
        return 32, 48

def _display_example_and_images():
    """
    Displays your example spreadsheet logic and plate reader layout image.
    """
    example_file_path = os.path.join("assets", "example_spreadsheet.xlsx")
    default_layout_image_path = os.path.join("assets", "image.png")

    col_left, col_right = st.columns(2)

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
            st.error("❌ Example file not found. Check the file path.")

        st.write("""
            Please ensure that your file follows this format:
            
            - The first column is `Time` with time points.
            - Subsequent columns are well measurements (A1, A2, B1, B2, etc.).
            """)
        sample_file = create_example_data()
        st.dataframe(sample_file)

    with col_right:
        st.subheader("Example of 96-well Plate Reader Layout")
        if os.path.exists(default_layout_image_path):
            default_image = Image.open(default_layout_image_path)
            max_width, max_height = 500, 500
            default_image.thumbnail((max_width, max_height))
            st.image(default_image, caption="Default Plate Reader Layout", use_container_width=False)
        else:
            st.info("⚠️ No default layout image available.")
