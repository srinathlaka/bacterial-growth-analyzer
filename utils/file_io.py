import streamlit as st
import pandas as pd
import numpy as np
import uuid
import os

from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from PIL import Image
import json

@st.cache_data
def read_data(uploaded_file, rows, columns):
    """
    Reads CSV or XLSX from Streamlit's file_uploader and adjusts layout.
    Automatically drops the first row if it looks like a header (e.g., contains 'Time').
    """
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, header=None)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
        else:
            st.warning("Check your file format: Only .csv or .xlsx files are supported.")
            return None
        # If the first cell is a string and matches 'Time' (case-insensitive), drop the first row
        if isinstance(df.iloc[0,0], str) and df.iloc[0,0].strip().lower() == 'time':
            st.warning("It looks like your file has a header row. Please upload files WITHOUT headers—just raw numeric data, with the first row as the first time point. Example files are available in the app.")
            df = df.iloc[1:].reset_index(drop=True)
        # Check for any column that is entirely non-numeric (e.g., text columns)
        non_numeric_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric_cols:
            st.warning(f"Your file contains non-numeric columns ({', '.join(str(c) for c in non_numeric_cols)}). This format is not supported. Please upload a numeric data file.")
            return None
        df = adjust_dataframe_layout(df, rows, columns)
        st.success("File uploaded successfully!")
        return df
    except Exception as e:
        st.warning("Check your file format. Unable to read the file as expected.")
        return None

def create_example_data():
    """
    Example data for display in Tab 1.
    """
    example_data={
        "Time":[0,1,2,3,4],
        "A1":[0.1,0.15,0.2,0.25,0.3],
        "A2":[0.2,0.25,0.3,0.35,0.4],
        "B1":[0.3,0.35,0.4,0.45,0.5]
    }
    return pd.DataFrame(example_data)

def adjust_dataframe_layout(df, rows, columns):
    total_columns = rows * columns
    if df.shape[1] < total_columns + 1:
        extra_cols = pd.DataFrame(np.nan, index=df.index, columns=range(df.shape[1], total_columns + 1))
        df = pd.concat([df, extra_cols], axis=1)

    df = df.iloc[:, :total_columns + 1]

    # Generate actual well labels like A1, A2, B1, B6, etc.
    well_labels = generate_labels(rows, columns)
    df.columns = ["Time"] + well_labels

    return df


def generate_labels(rows, cols):
    labels = []
    for i in range(rows):
        row_letter = chr(ord('A') + i)  # e.g. 'A' if i=0, 'B' if i=1...
        for j in range(cols):
            label = f"{row_letter}{j+1}"
            labels.append(label)
    return labels


def create_button_layout(rows, columns, labels, key_prefix, df=None):
    """
    The advanced well selection UI with colored buttons for selected wells.
    """
    st.markdown("""
    <style>
      .block-container { padding-top:0rem; padding-bottom:0rem; }
      .stButton,.stRadio { margin:0rem; padding:0rem; }
      div[data-testid="column"] { padding:0rem 0.2rem; }
      
      /* Style for selected wells (primary) */
      .stButton button[data-testid="baseButton-primary"] {
        background-color: #d1e7dd;
        border-color: #badbcc;
        color: #0f5132;
      }
    </style>
    """, unsafe_allow_html=True)

    for k in ["selected_wells","selected_rows","selected_cols","manual_deselected"]:
        if f"{key_prefix}_{k}" not in st.session_state:
            st.session_state[f"{key_prefix}_{k}"]=set()

    select_all=st.checkbox("✅ Select All Wells", key=f"{key_prefix}_select_all")
    if select_all:
        st.session_state[f"{key_prefix}_selected_wells"]=set(labels)
        st.session_state[f"{key_prefix}_selected_rows"].clear()
        st.session_state[f"{key_prefix}_selected_cols"].clear()
        st.session_state[f"{key_prefix}_manual_deselected"].clear()

    top_row=st.columns(columns+1, gap="small")
    top_row[0].write(" ")
    for j in range(columns):
        col_key = f"{key_prefix}_col_select_{j}"
        # Default to first option if reset flag is set
        default_index = 0 if st.session_state.get(f"{key_prefix}_reset_radios", False) else None
        top_row[j+1].radio(" ", [" ", "✓"], index=default_index, key=col_key, label_visibility="collapsed")

    # After creating all radio buttons, clear the reset flag
    if st.session_state.get(f"{key_prefix}_reset_radios", False):
        st.session_state[f"{key_prefix}_reset_radios"] = False

    # Row-column selection - use horizontal layout for the buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        apply_selection=st.button("Select by Row/Column", key=f"{key_prefix}_apply_selection")

    for i in range(rows):
        row_layout=st.columns(columns+1, gap="small")
        row_key=f"{key_prefix}_row_select_{i}"
        default_index = 0 if st.session_state.get(f"{key_prefix}_reset_radios", False) else None
        row_layout[0].radio(" ",[" ","✓"],index=default_index,key=row_key,label_visibility="collapsed")
        row_layout[0].write(f"**{chr(65+i)}**")
        for j in range(columns):
            idx=i*columns+j
            well_name=labels[idx]
            is_selected=(well_name in st.session_state[f"{key_prefix}_selected_wells"])
            
            # Use primary type for selected wells, secondary for unselected
            button_type = "primary" if is_selected else "secondary"
            if row_layout[j+1].button(well_name, key=f"{key_prefix}_{well_name}", type=button_type):
                if is_selected:
                    st.session_state[f"{key_prefix}_selected_wells"].remove(well_name)
                    st.session_state[f"{key_prefix}_manual_deselected"].add(well_name)
                else:
                    st.session_state[f"{key_prefix}_selected_wells"].add(well_name)
                    st.session_state[f"{key_prefix}_manual_deselected"].discard(well_name)
                st.rerun()  # Add this to update UI immediately

    if apply_selection:
        selected_cols_1b={j+1 for j in range(columns)
                          if st.session_state.get(f"{key_prefix}_col_select_{j}"," ")=="✓"}
        selected_rows_1b={i+1 for i in range(rows)
                          if st.session_state.get(f"{key_prefix}_row_select_{i}"," ")=="✓"}
        st.session_state[f"{key_prefix}_selected_cols"]=selected_cols_1b
        st.session_state[f"{key_prefix}_selected_rows"]=selected_rows_1b

        max_col_1b=max(selected_cols_1b) if selected_cols_1b else -1
        max_row_1b=max(selected_rows_1b) if selected_rows_1b else -1

        idx=0
        for r in range(rows):
            row_1b=r+1
            for c in range(columns):
                col_1b=c+1
                well_name=labels[idx]
                idx+=1
                manually_deselected=(well_name in st.session_state[f"{key_prefix}_manual_deselected"])
                manually_selected=(well_name in st.session_state[f"{key_prefix}_selected_wells"])
                if selected_rows_1b and not selected_cols_1b:
                    should_select=(row_1b in selected_rows_1b)
                elif selected_cols_1b and not selected_rows_1b:
                    should_select=(col_1b in selected_cols_1b)
                else:
                    should_select=(
                        (row_1b in selected_rows_1b and col_1b<=max_col_1b) or
                        (col_1b in selected_cols_1b and row_1b<=max_row_1b)
                    )
                if should_select and not manually_deselected:
                    st.session_state[f"{key_prefix}_selected_wells"].add(well_name)
                elif not manually_selected:
                    st.session_state[f"{key_prefix}_selected_wells"].discard(well_name)
        st.success("✅ Wells selected based on row/column intersection!")
        st.rerun()  # Add this line to refresh the UI

    # Action buttons - use horizontal layout
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear Selection", key=f"{key_prefix}_clear"):
            if select_all:
                st.warning("⚠️ Please uncheck 'Select All Wells' first.")
            else:
                # Only clear the selections, don't try to modify widget values directly
                st.session_state[f"{key_prefix}_manual_deselected"].clear()
                st.session_state[f"{key_prefix}_selected_wells"].clear()
                st.session_state[f"{key_prefix}_selected_rows"].clear()
                st.session_state[f"{key_prefix}_selected_cols"].clear()
                
                # Add a flag to indicate we want to reset the radio buttons
                st.session_state[f"{key_prefix}_reset_radios"] = True
                
                st.success("✅ Selection cleared!")
                st.rerun()  # This will cause the app to re-render with cleared selections
    
    # New Plot Selected button
    with col2:
        plot_selected = st.button("📊 Plot Selected", key=f"{key_prefix}_plot_selected")

    selected_wells=sorted(st.session_state[f"{key_prefix}_selected_wells"])
    if selected_wells:
        st.markdown("### ✅ Selected Wells")
        st.markdown(" ".join([
            f"<span style='padding:6px 12px; margin:4px; background:#d1e7dd; border-radius:20px;'>{w}</span>"
            for w in selected_wells
        ]),unsafe_allow_html=True)
    else:
        st.info("No wells selected.")
    
    # Return both selected wells and the plot flag
    return selected_wells, plot_selected

def create_custom_model(custom_expr, param_names):
    """
    Create a callable custom model from a user-defined expression, e.g. "X0*exp(mu*t)".
    """
    if not custom_expr:
        st.error("Custom model expression is empty.")
        return None
    try:
        import sympy as sp
        t=sp.symbols('t')
        params=sp.symbols(param_names)
        expr=sp.sympify(custom_expr)
        func=sp.lambdify([t]+list(params), expr, 'numpy')
        return func
    except Exception as e:
        st.error(f"Error parsing custom model expression: {e}")
        return None
