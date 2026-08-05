import streamlit as st
import os

# Bump this on each release; shown in the footer.
LAST_UPDATED = "2026-08-05"

from tabs import (
    tab_upload,
    tab_background,
    tab_operations,
    tab_fitting,
    tab_phase_detection,
    tab_growth_models
)

# Only import tab_ode_analysis if the file exists
try:
    from tabs import tab_ode_analysis
except ImportError:
    tab_ode_analysis = None
    # Optionally: st.warning("ODE Analysis tab is not available.")

def main():
    st.set_page_config(
        page_title="Bacterial Growth Analyzer",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add CSS for fixed header with improved visibility
    st.markdown("""
    <style>
        .fixed-header {
            position: sticky;
            top: 0;
            background-color: #203864;  /* Dark blue background for better contrast */
            z-index: 999;
            padding: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);  /* Add shadow for depth */
            width: 100%;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create a fixed header with app title with better visibility
    st.markdown("""
    <div class="fixed-header">
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: white; margin: 0; font-weight: bold; font-size: 24px;">
                <img src="https://img.icons8.com/color/48/000000/bacteria.png" style="height: 30px; margin-right: 10px; vertical-align: bottom;">
                Bacterial Growth Analyzer
            </h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add a spacer after the fixed header
    st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .mode-card {
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            transition: transform 0.3s;
            cursor: pointer;
        }
        .mode-card:hover {
            transform: scale(1.02);
        }
        .basic-mode {
            background-color: #f0f7ff;
            border: 2px solid #4d94ff;
        }
        .advanced-mode {
            background-color: #fff0f7;
            border: 2px solid #ff4d94;
        }
        .active-mode {
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.5);
            transform: scale(1.05);
        }
        .progress-tracker {
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .step-complete {
            color: #28a745;
            font-weight: bold;
        }
        .step-current {
            color: #007bff;
            font-weight: bold;
        }
        .step-pending {
            color: #6c757d;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize user level if not set
    if "user_level" not in st.session_state:
        st.session_state["user_level"] = "Basic"
    
    # Add this initialization near the beginning of the main() function, where you initialize other session variables
    if "sample_wells_by_group" not in st.session_state:
        st.session_state["sample_wells_by_group"] = {}
    if "blank_wells_by_group" not in st.session_state:
        st.session_state["blank_wells_by_group"] = {}
    if "groups_data" not in st.session_state:
        st.session_state["groups_data"] = {}
    if "selected_sample_wells_by_group" not in st.session_state:
        st.session_state["selected_sample_wells_by_group"] = {}
    
    # Create tabs dictionary
    tabs = {
        "Upload": {"title": "📂 Upload Data", "function": tab_upload.display_tab_upload, "icon": "📂", "order": 1},
        "Background": {"title": "🧪 Background", "function": tab_background.display_tab_background, "icon": "🧪", "order": 2},
        "Operations": {"title": "🔧 Operations", "function": tab_operations.display_tab_operations, "icon": "🔧", "order": 3},
        "Fitting": {"title": "📈 Fitting", "function": tab_fitting.display_tab_fitting, "icon": "📈", "order": 4},
        "ODE": {"title": "🔬 ODE Analysis", "function": tab_ode_analysis.display_tab_ode_analysis, "icon": "🔬", "order": 5},
        "Phase Detection": {"title": "🔍 Phase Detection", "function": tab_phase_detection.display_tab_phase_detection, "icon": "🔍", "order": 6},
        "Growth Models": {"title": "📚 Models Info", "function": tab_growth_models.display_tab_growth_models, "icon": "📚", "order": 7}
    }
    
    # Sidebar with visual style
    with st.sidebar:
        st.title("🦠 Bacterial Growth Analyzer")
        
        # User mode selector with attractive cards
        st.write("### User Mode")
        col1, col2 = st.columns(2)
        
        basic_active = st.session_state["user_level"] == "Basic"
        advanced_active = not basic_active
        
        with col1:
            basic_class = "mode-card basic-mode" + (" active-mode" if basic_active else "")
            st.markdown(f"""
            <div class="{basic_class}" onclick="this.querySelector('button').click();">
                <h3>Basic</h3>
                <p>Simplified workflow</p>
                <p>🔹 For beginners</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Basic", key="select_basic", help="Simplified view with essential features"):
                # Store important session state variables before changing mode
                preserve_keys = ["df", "rows", "columns", "upload_tab_selected_wells", 
                                 "groups_data", "operated_data", "selected_operated_wells",
                                 "num_groups", "selected_sample_wells_by_group"]
                
                # Create a temporary dictionary to store values
                temp_storage = {}
                for key in preserve_keys:
                    if key in st.session_state:
                        temp_storage[key] = st.session_state[key]
                
                # Change the mode
                st.session_state["user_level"] = "Basic"
                
                # Check if current tab is valid in Basic mode
                basic_tabs = ["Upload", "Background", "Fitting", "Growth Models"]
                current_tab = st.session_state.get("current_tab", "Upload")
                if current_tab not in basic_tabs:
                    # Reset to Upload tab if current tab is not available in Basic mode
                    st.session_state["current_tab"] = "Upload"
                
                # Restore preserved values
                for key, value in temp_storage.items():
                    st.session_state[key] = value
                
                st.rerun()
                
        with col2:
            advanced_class = "mode-card advanced-mode" + (" active-mode" if advanced_active else "")
            st.markdown(f"""
            <div class="{advanced_class}" onclick="this.querySelector('button').click();">
                <h3>Advanced</h3>
                <p>Full features</p>
                <p>🔸 For experts</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Advanced", key="select_advanced", help="Access all features and customizations"):
                st.session_state["user_level"] = "Advanced"
                st.rerun()
        
        # Visual workflow tracker
        st.write("### Your Progress")
        current_tab = st.session_state.get("current_tab", "Upload")
        
        # Custom progress tracker based on user level
        if st.session_state["user_level"] == "Basic":
            basic_tabs = ["Upload", "Background", "Fitting", "Growth Models"]
            progress_html = '<div class="progress-tracker">'
            
            for i, tab in enumerate(basic_tabs):
                if tab == current_tab:
                    status = "step-current"
                    marker = "▶️"
                elif i < basic_tabs.index(current_tab):
                    status = "step-complete" 
                    marker = "✅"
                else:
                    status = "step-pending"
                    marker = "⏳"
                    
                progress_html += f'<div class="{status}">{marker} {i+1}. {tabs[tab]["icon"]} {tab}</div>'
            
            progress_html += '</div>'
            st.markdown(progress_html, unsafe_allow_html=True)
            
            # Navigation
            st.write("### Navigation")
            for tab in basic_tabs:
                if st.button(tabs[tab]["title"], key=f"tab_{tab}"):
                    st.session_state["current_tab"] = tab
                    st.rerun()
        else:
            # Advanced mode - show all tabs with progress
            progress_html = '<div class="progress-tracker">'
            all_tabs = sorted([(name, info) for name, info in tabs.items()], key=lambda x: x[1]["order"])
            
            for i, (tab_name, tab_info) in enumerate(all_tabs):
                if tab_name == current_tab:
                    status = "step-current"
                    marker = "▶️"
                elif i < [t[0] for t in all_tabs].index(current_tab):
                    status = "step-complete" 
                    marker = "✅"
                else:
                    status = "step-pending"
                    marker = "⏳"
                    
                progress_html += f'<div class="{status}">{marker} {i+1}. {tab_info["icon"]} {tab_name}</div>'
            
            progress_html += '</div>'
            st.markdown(progress_html, unsafe_allow_html=True)
            
            # Navigation
            st.write("### Navigation")
            for tab_name, tab_info in all_tabs:
                if st.button(tab_info["title"], key=f"tab_{tab_name}"):
                    st.session_state["current_tab"] = tab_name
                    st.rerun()
    
    # Display the current tab with header based on mode
    current_tab = st.session_state.get("current_tab", "Upload")
    
    mode_type = "Basic" if st.session_state["user_level"] == "Basic" else "Advanced"
    mode_icon = "🔷" if st.session_state["user_level"] == "Basic" else "🔶"
    mode_color = "#4d94ff" if st.session_state["user_level"] == "Basic" else "#ff4d94"

    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 15px;">
        <div style="background-color: {mode_color}20; border: 2px solid {mode_color}; 
             border-radius: 20px; padding: 5px 15px; display: inline-block;">
            <span style="font-size: 1.2em; font-weight: bold; color: {mode_color};">
                {mode_icon} {mode_type} Mode
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Call the appropriate tab function
    tabs[current_tab]["function"]()

    # Add data bridge for basic users
    if st.session_state.get("user_level") == "Basic" and current_tab == "Background":
        # For the bridge between Background and Fitting tabs
        if 'groups_data' in st.session_state and st.session_state['groups_data']:
            group1_data = st.session_state['groups_data'].get("Group_1_bg_subtracted")
            sample_wells_group1 = st.session_state.get("selected_sample_wells_by_group", {}).get(1, [])
            
            if group1_data is not None and sample_wells_group1:
                st.success("✅ Background subtraction complete!")
                
                # Automatically create operated_data for fitting tab to use
                st.session_state["operated_data"] = group1_data
                st.session_state["selected_operated_wells"] = sample_wells_group1
                
                # Add a prominent continue button
                st.markdown("""
                <div style="text-align: center; margin-top: 30px; margin-bottom: 30px;">
                    <h3>Ready for Fitting!</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("▶️ Continue to Fitting", key="continue_to_fitting"):
                    st.session_state["current_tab"] = "Fitting"
                    st.rerun()

    # Footer
    st.markdown("<hr style='margin-top:40px; margin-bottom:10px;'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center; color:#6c757d; font-size:0.85em; padding-bottom:10px;'>"
        f"Last updated: {LAST_UPDATED}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
