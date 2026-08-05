import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import uuid

def plot_all_ode_fits_summary(ode_fits, operated_data, selected_operated_wells):
    """
    Summarize multiple ODE fits in one figure with the overall average data.
    (From lines ~130–185).
    """
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=operated_data["Time"],
        y=operated_data["Average"],
        mode='lines',
        name='Overall Average',
        line=dict(color='black',width=2,dash='dot')
    ))
    for i,fit_item in enumerate(ode_fits):
        fit_results=fit_item.get("fit_results")
        if not fit_results:
            continue
        fit_t=fit_results.get("fit_time")
        sol_y=fit_results.get("solution_y")
        if fit_t is None or sol_y is None:
            continue
        fig.add_trace(go.Scatter(
            x=fit_t,
            y=sol_y[0],
            mode='lines',
            name=f"Fit {i+1} X",
            line=dict(width=2,color='green')
        ))
        if sol_y.shape[0]>1:
            fig.add_trace(go.Scatter(
                x=fit_t,
                y=sol_y[1],
                mode='lines',
                name=f"Fit {i+1} Y",
                line=dict(width=2,color='red',dash='dash')
            ))
        lower=fit_results.get("lower_bound")
        upper=fit_results.get("upper_bound")
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([fit_t, fit_t[::-1]]),
                y=np.concatenate([upper,lower[::-1]]),
                fill='toself',
                fillcolor='rgba(173,216,230,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name=f"Fit {i+1} X 95% CI",
                showlegend=False
            ))
        params=fit_results.get("parameters")
        param_names=fit_item.get("parameters")
        if params is not None and param_names is not None:
            annotation_text=", ".join([f"{name}={val:.3f}" for name,val in zip(param_names,params)])
            t_mid=fit_t[len(fit_t)//2]
            y_mid=np.max(sol_y[0])
            fig.add_annotation(x=t_mid,y=y_mid,text=annotation_text,showarrow=True,arrowhead=1)
    fig.update_layout(
        title="Summary: All ODE Fits (X & Y)",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_white"
    )
    return fig

def plot_avg_sd_operated(data, selected_wells):
    """
    Plot average & std of the 'operated_data' DataFrame. 
    Lines ~1150 from your code.
    """
    avg=data["Average"]
    sd=data[selected_wells].std(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Time"],
        y=avg,
        mode='lines',
        name='Average',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data["Time"],
        y=avg+sd,
        mode='lines',
        name='Average+SD',
        line=dict(color='lightblue'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=data["Time"],
        y=avg-sd,
        mode='lines',
        name='Average-SD',
        line=dict(color='lightblue'),
        fill='tonexty',
        fillcolor='rgba(173,216,230,0.3)',
        showlegend=False
    ))
    fig.update_layout(
        title="Average and Standard Deviation of Operated Data",
        xaxis_title="Time",
        yaxis_title="OD",
        template='plotly_white'
    )
    unique_key=f"plot_avg_sd_operated_{uuid.uuid4().hex}"
    st.plotly_chart(fig,use_container_width=True,key=unique_key)

def plot_selected_wells(df, wells, context="raw"):
    """Plot selected wells."""
    # Convert wells to a list if it's a pandas Index or Series
    if isinstance(wells, (pd.Index, pd.Series)):
        wells = wells.tolist()
    
    if not wells:
        st.warning("No wells selected to plot.")
        return
    
    # Create a stable key for the plot that includes context
    wells_key = "_".join(sorted(wells[:3])) if wells else "empty"  # use up to first 3 wells for key
    plot_key = f"selected_wells_{context}_{wells_key}_{len(wells)}"
    
    fig = go.Figure()
    for well in wells:
        if well in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Time'], 
                y=df[well],
                mode='lines',
                name=well
            ))
    
    fig.update_layout(
        title='Selected Wells',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=plot_key)

def plot_confidence_intervals(df, lower_bound, upper_bound, y_pred, std_dev, fit_id=None):
    """Plot confidence intervals & standard deviation for a fitted curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([df['Time'], df['Time'][::-1]]),
        y=np.concatenate([lower_bound, upper_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(65,105,225,0.35)',
        line=dict(color='rgba(65,105,225,0.8)', width=1),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Average'],
        mode='lines',
        name='Observed Data',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=df['Time'].tolist()+df['Time'].tolist()[::-1],
        y=(df['Average']-std_dev).tolist()+(df['Average']+std_dev).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(144,238,144,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Standard Deviation'
    ))
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=y_pred,
        mode='lines',
        name='Fitted Curve',
        line=dict(color='crimson')
    ))
    fig.update_layout(
        title='Confidence Intervals with Standard Deviation and Fitted Curve',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    # Generate a stable key
    if fit_id is None:
        fit_id = hash(str(lower_bound[0]) + str(upper_bound[0])) % 10000
        
    plot_key = f"confidence_intervals_plot_{fit_id}"
    st.plotly_chart(fig, key=plot_key)

# Update the plot_fitted_curves function to use stable keys
def plot_fitted_curves(df, time, observed, fitted, model_name, fit_id=None):
    """Plot a simple observed vs fitted curve for a particular model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=observed,mode='lines',name='Observed Data'))
    fig.add_trace(go.Scatter(x=time,y=fitted,mode='lines',name=f'Fitted Curve ({model_name})',line=dict(color='red')))
    fig.update_layout(
        title=f'Fitted {model_name} Model',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    # Generate a stable key based on the model name and a unique ID
    if fit_id is None:
        fit_id = hash(str(time[0]) + str(time[-1])) % 10000  # Create ID from time range
    
    plot_key = f"fitted_curve_{model_name}_{fit_id}"
    st.plotly_chart(fig, key=plot_key)

def plot_phase_fit_with_ci(phase_fits, operated_data, selected_operated_wells):
    """
    Plot all phase fits with confidence intervals on a single plot
    """
    fig = go.Figure()
    
    # Add observed data
    fig.add_trace(go.Scatter(
        x=operated_data["Time"],
        y=operated_data["Average"],
        mode='lines',
        name='Average Data',
        line=dict(color='black', width=2)
    ))
    
    # Add individual fits
    for phase_num, phase in enumerate(phase_fits):
        if not phase.get("fit_results"):
            continue
            
        # Get fit data from phase
        phase_time = phase.get("phase_time", [])
        y_pred = phase.get("fit", [])
        lower_bound = phase.get("lower_bound", [])
        upper_bound = phase.get("upper_bound", [])
        std_dev = phase.get("std_dev", [])
        
        if not len(phase_time) or not len(y_pred):
            continue
            
        # Add fitted curve
        fig.add_trace(go.Scatter(
            x=phase_time,
            y=y_pred,
            mode='lines',
            name=f'Phase {phase_num+1} Fit',
            line=dict(width=2)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=phase_time.tolist()+phase_time.tolist()[::-1],
            y=upper_bound.tolist()+lower_bound.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f'Phase {phase_num+1} CI',
            showlegend=False
        ))
        
        # Add standard deviation
        upper_sd = (y_pred+std_dev).tolist()
        lower_sd = (y_pred-std_dev).tolist()
        fig.add_trace(go.Scatter(
            x=phase_time.tolist()+phase_time.tolist()[::-1],
            y=upper_sd+lower_sd[::-1],
            fill='toself',
            fillcolor='rgba(144,238,144,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f'Phase {phase_num+1} Std Dev',
            showlegend=False
        ))
    
    fig.update_layout(
        title='All Phase Fits with Confidence Intervals and Standard Deviations',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    
    return fig

def display_single_well_preview(df, well_name):
    """
    Display a preview plot for a single well.
    """
    if (well_name and well_name in df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df[well_name],
            mode='lines',
            name=well_name
        ))
        fig.update_layout(
            title=f'Preview of {well_name}',
            xaxis_title='Time',
            yaxis_title='OD',
            template='plotly_white',
            width=400,
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, key=f"preview_{well_name}_{uuid.uuid4().hex}")
    else:
        st.write("No well selected for preview.")


def plot_avg_sd_bg_subtracted(data, wells, group_num, context="default"):
    """
    Plots the average and standard deviation of background-subtracted data.
    """
    # Filter wells to only include those that exist in the data
    valid_wells = [well for well in wells if well in data.columns]
    
    if not valid_wells:
        st.warning(f"None of the selected wells exist in the data for group {group_num}")
        return

    # Calculate average and standard deviation
    avg_data = data[valid_wells].mean(axis=1)
    std_data = data[valid_wells].std(axis=1)

    # Create the figure
    fig = go.Figure()

    # CRITICAL FIX: Use data['Time'] as x-axis instead of data.index
    # Add average line
    fig.add_trace(go.Scatter(
        x=data['Time'],  # Changed from data.index to data['Time']
        y=avg_data,
        mode='lines',
        name=f'Group {group_num} Average'
    ))

    # Add shaded area for standard deviation
    # FIX: Use Time column values for x-axis here too
    fig.add_trace(go.Scatter(
        x=list(data['Time']) + list(data['Time'][::-1]),  # Use Time column
        y=list(avg_data + std_data) + list((avg_data - std_data)[::-1]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'Group {group_num} Std Dev'
    ))

    # Update layout
    fig.update_layout(
        title=f'Background-Subtracted Data (Group {group_num})',
        xaxis_title='Time',
        yaxis_title='Optical Density (OD)',
        template='plotly_white'
    )

    # Create a unique key that doesn't depend on specific well names
    # Use a hash of the valid wells rather than the well names themselves
    wells_hash = hash(tuple(sorted(valid_wells))) % 10000  # Use modulo to keep it reasonable size
    plot_key = f"plot_avg_sd_group_{group_num}_{context}_{wells_hash}"
    
    st.session_state[f"plot_key_group_{group_num}_{context}"] = plot_key
    st.plotly_chart(fig, key=plot_key)

def plot_raw_vs_corrected(df, group_df, well_name, group_num):
    """Compare raw and background-corrected data for a single well"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Time'], 
        y=df[well_name], 
        mode='lines', 
        name='Raw Data',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=group_df['Time'], 
        y=group_df[well_name], 
        mode='lines', 
        name='Background Corrected',
        line=dict(color='green')
    ))
    fig.update_layout(
        title=f'Raw vs. Background-Corrected Data - {well_name} (Group {group_num})',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    return fig

def plot_average_blank(df, blank_wells):
    """Plot average of blank wells."""
    if not blank_wells:
        st.warning("No blank wells selected.")
        return
    
    # Create a stable key for the plot
    wells_key = "_".join(sorted(blank_wells)[:3])
    plot_key = f"blank_wells_{wells_key}_{len(blank_wells)}"
    
    fig = go.Figure()
    
    # Plot individual blank wells
    for well in blank_wells:
        if well in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Time'],
                y=df[well],
                mode='lines',
                name=f'Blank {well}',
                opacity=0.5,
                line=dict(width=1)
            ))
    
    # Plot average of blank wells
    avg_blank = df[blank_wells].mean(axis=1)
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=avg_blank,
        mode='lines',
        name='Average of Blanks',
        line=dict(color='black', width=2)
    ))
    
    fig.update_layout(
        title='Blank Wells Data',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=plot_key)

def plot_blank_fit(df, avg_blank, y_pred, lower_bound, upper_bound, group_num):
    """
    Plot the blank well data with fitted model and confidence intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing Time column and blank well data
    avg_blank : pandas.Series
        The average values of blank wells
    y_pred : numpy.ndarray
        The fitted model predictions
    lower_bound : numpy.ndarray
        Lower confidence interval boundary
    upper_bound : numpy.ndarray
        Upper confidence interval boundary
    group_num : int
        Group number for display purposes
        
    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The plotly figure object ready to be displayed
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add the raw data
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=avg_blank,
        mode='markers',
        name='Average of Blank Wells',
        marker=dict(color='blue', size=8)
    ))
    
    # Add the fitted curve
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=y_pred,
        mode='lines',
        name='Fitted Model',
        line=dict(color='red', width=2)
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=lower_bound,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name='95% Confidence Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Blank Wells Fitting - Group {group_num}',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig