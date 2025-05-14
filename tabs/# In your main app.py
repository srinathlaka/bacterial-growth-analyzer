# In your main app.py file, modify the mode selection section
def main():
    # ... existing code ...
    
    # App mode selection
    app_mode = st.sidebar.selectbox(
        "App Mode",
        ["Basic", "Advanced"],
        index=0
    )
    
    # Set number of groups based on mode
    if app_mode == "Basic":
        # Force single group in basic mode
        num_groups = 1
        st.session_state.num_groups = 1
    else:
        # Allow group selection in advanced mode
        num_groups = st.sidebar.number_input(
            "Number of Groups",
            min_value=1,
            max_value=10,
            value=st.session_state.get("num_groups", 1)
        )
        st.session_state.num_groups = num_groups