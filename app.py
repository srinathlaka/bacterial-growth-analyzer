import streamlit as st
import os
from tabs import (
    tab_upload,
    tab_background,
    tab_operations,
    tab_fitting,
    tab_ode_analysis,
    tab_phase_detection,
    tab_growth_models
)

def main():
    """
    Main entry point that creates seven tabs for the bacterial growth app.
    """
    st.set_page_config(page_title="Bacterial Growth Analysis Toolbox", page_icon="🔬", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bacterial Growth Analysis Toolbox</h1>", unsafe_allow_html=True)

    # Create seven tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Upload Data",
        "Background Subtraction",
        "Operations",
        "Fitting",
        "Custom ODE Analysis",
        "Automatic Phase Detection",
        "Growth Models"
    ])

    with tab1:
        tab_upload.display_tab_upload()

    with tab2:
        tab_background.display_tab_background()

    with tab3:
        tab_operations.display_tab_operations()

    with tab4:
        tab_fitting.display_tab_fitting()

    with tab5:
        tab_ode_analysis.display_tab_ode_analysis()

    with tab6:
        tab_phase_detection.display_tab_phase_detection()

    with tab7:
        tab_growth_models.display_tab_growth_models()

if __name__ == "__main__":
    main()
