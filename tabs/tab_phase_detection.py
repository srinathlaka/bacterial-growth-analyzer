import streamlit as st
import numpy as np
from utils.phase_detection import detect_phases, plot_detected_phases
from utils.plotting import plot_avg_sd_operated

def display_tab_phase_detection():
    """
    Tab 6: Automatic Phase Detection.
    Using thresholded_derivative or slope_based from phase_detection.py
    """
    st.header("🔍 Automatic Phase Detection (Merged)")

    operated_data=st.session_state.get("operated_data")
    selected_operated_wells=st.session_state.get("selected_operated_wells",[])

    if operated_data is None or not selected_operated_wells:
        st.warning("⚠️ Please perform operations in previous tabs first.")
        return

    if "Average" not in operated_data.columns:
        operated_data["Average"]=operated_data[selected_operated_wells].mean(axis=1)

    method=st.selectbox(
        "Select Phase Detection Method",
        ["thresholded_derivative","slope_based"],
        format_func=lambda m: "Thresholded Derivative" if m=="thresholded_derivative" else "Slope-Based"
    )

    time=operated_data["Time"].values
    od_values=operated_data["Average"].values

    if method=="thresholded_derivative":
        st.subheader("Thresholded Derivative Parameters")
        from math import floor
        smoothing_window=st.slider("Smoothing Window",3,51,5,step=2)
        polyorder=st.slider("Polynomial Order (Savitzky-Golay)",1,5,2)
        derivative_threshold=st.number_input("Derivative Threshold", value=0.005, step=0.001, format="%.5f")
        min_distance=st.slider("Min Distance Between Breakpoints",1,50,5)

        if st.button("Detect Phases"):
            phases, breakpoints, extra = detect_phases(
                time=time,
                od_values=od_values,
                method="thresholded_derivative",
                smoothing_window=smoothing_window,
                polyorder=polyorder,
                derivative_threshold=derivative_threshold,
                min_distance=min_distance
            )
            if not phases:
                st.error("No phases detected. Try adjusting parameters.")
            else:
                st.success(f"Detected {len(phases)} phase(s).")
                plot_detected_phases(time, od_values, phases,
                                     derivative=extra.get("derivative"),
                                     slopes=None,
                                     change_points=breakpoints)
                st.subheader("Detected Phases")
                phase_table=[]
                for idx,(start,end) in enumerate(phases):
                    phase_table.append({
                        "Phase":idx+1,
                        "Start":start,
                        "End":end,
                        "Duration":end-start
                    })
                st.dataframe(phase_table)

    else:
        st.subheader("Slope-Based Parameters")
        slope_window=st.slider("Slope Window Size",3,51,5,step=2)
        slope_threshold=st.number_input("Slope Threshold",value=0.001,step=0.0001,format="%.5f")

        if st.button("Detect Phases"):
            phases, breakpoints, extra = detect_phases(
                time=time,
                od_values=od_values,
                method="slope_based",
                slope_window=slope_window,
                slope_threshold=slope_threshold
            )
            if not phases:
                st.error("No phases detected. Try adjusting parameters.")
            else:
                st.success(f"Detected {len(phases)} phase(s).")
                plot_detected_phases(time, od_values, phases,
                                     derivative=None,
                                     slopes=extra.get("slopes"),
                                     change_points=breakpoints)
                st.subheader("Detected Phases")
                phase_table=[]
                for idx,(start,end) in enumerate(phases):
                    phase_table.append({
                        "Phase":idx+1,
                        "Start":start,
                        "End":end,
                        "Duration":end-start
                    })
                st.dataframe(phase_table)
