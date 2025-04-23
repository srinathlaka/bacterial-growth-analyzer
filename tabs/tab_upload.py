import streamlit as st
import os
from PIL import Image
import json

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
    st.header("📁 Raw Data Plotting")

    # Layout selection
    rows, columns = _select_layout()
    st.session_state["rows"] = rows
    st.session_state["columns"] = columns
    labels = generate_labels(rows, columns)

    # Display example and images if exist
    _display_example_and_images()

    st.subheader("📂 Upload Your Data File (CSV or XLSX)")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx","csv"], key="data_file")

    df = None
    if uploaded_file is not None:
        with st.spinner("Reading and processing the data..."):
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
