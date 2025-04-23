import streamlit as st
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

def plot_selected_wells(df, selected_wells):
    """
    Quick function to line-plot selected wells from DataFrame. 
    Lines ~1030 or so. 
    """
    fig=go.Figure()
    for well in selected_wells:
        if well not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=df["Time"], y=df[well], mode='lines', name=well))
    fig.update_layout(
        title="Time vs Selected Wells",
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    unique_key=f"plot_selected_wells_{'_'.join(selected_wells)}_{uuid.uuid4().hex}"
    st.plotly_chart(fig,use_container_width=True,key=unique_key)

def plot_confidence_intervals(df, lower_bound, upper_bound, y_pred, std_dev):
    """
    Plot confidence intervals & standard dev for a fitted curve. 
    Lines ~1034–1100 snippet from your code.
    """
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([df['Time'], df['Time'][::-1]]),
        y=np.concatenate([lower_bound, upper_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.4)',
        line=dict(color='rgba(255,255,255,0)'),
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
    st.plotly_chart(fig)

def plot_fitted_curves(df, time, observed, fitted, model_name):
    """
    Plot a simple observed vs fitted curve for a particular model. 
    Lines ~760+ from your code.
    """
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time,y=observed,mode='lines',name='Observed Data'))
    fig.add_trace(go.Scatter(x=time,y=fitted,mode='lines',name=f'Fitted Curve ({model_name})',line=dict(color='red')))
    fig.update_layout(
        title=f'Fitted {model_name} Model',
        xaxis_title='Time',
        yaxis_title='OD',
        legend_title='Legend',
        template='plotly_white'
    )
    st.plotly_chart(fig)

def plot_phase_fit_with_ci(phase_fits, operated_data, selected_operated_wells):
    """
    Summarize multiple phase-based fits with confidence intervals. 
    Lines ~~ ??? from your code. 
    """
    fig=go.Figure()
    average_values=operated_data[selected_operated_wells].mean(axis=1)
    fig.add_trace(go.Scatter(
        x=operated_data["Time"],
        y=average_values,
        mode='lines',
        name='Average Data',
        line=dict(color='black',width=2)
    ))
    for fit in phase_fits:
        phase_num=fit['phase']
        model_name=fit['model']
        phase_time=fit['phase_time']
        y_pred=fit['fit']
        lower=fit['lower_bound']
        upper=fit['upper_bound']
        std_dev=fit['std_dev']

        fig.add_trace(go.Scatter(
            x=np.concatenate([phase_time,phase_time[::-1]]),
            y=np.concatenate([lower,upper[::-1]]),
            fill='toself',
            fillcolor='rgba(173,216,230,0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f'Phase {phase_num} 95% CI',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=phase_time,
            y=y_pred,
            mode='lines',
            name=f'Phase {phase_num} Fit ({model_name})',
            line=dict(width=2)
        ))
        upper_sd=(y_pred+std_dev).tolist()
        lower_sd=(y_pred-std_dev).tolist()
        fig.add_trace(go.Scatter(
            x=phase_time.tolist()+phase_time.tolist()[::-1],
            y=upper_sd+lower_sd[::-1],
            fill='toself',
            fillcolor='rgba(144,238,144,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f'Phase {phase_num} Std Dev',
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
    if well_name and well_name in df.columns:
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


def plot_avg_sd_bg_subtracted(data, wells, group_num):
    """
    Plots the average and standard deviation of background-subtracted data.

    Parameters:
    - data: DataFrame containing the background-subtracted data.
    - wells: List of wells to include in the plot.
    - group_num: Integer representing the group number (used for unique keys).
    """
    # Calculate average and standard deviation
    avg_data = data[wells].mean(axis=1)
    std_data = data[wells].std(axis=1)

    # Create the figure
    fig = go.Figure()

    # Add average line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=avg_data,
        mode='lines',
        name=f'Group {group_num} Average'
    ))

    # Add shaded area for standard deviation
    fig.add_trace(go.Scatter(
        x=list(data.index) + list(data.index[::-1]),
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

    # Generate a truly unique key using uuid
    unique_key = f"plot_avg_sd_group_{group_num}_{uuid.uuid4()}"
    st.plotly_chart(fig, key=unique_key)

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
    """Plot the average of blank wells"""
    import plotly.graph_objects as go
    import uuid
    
    fig = go.Figure()
    
    # First plot each individual blank well
    for well in blank_wells:
        if well in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Time'],
                y=df[well],
                mode='lines',
                opacity=0.3,
                name=f'Blank: {well}'
            ))
    
    # Then plot the average of all blank wells
    if blank_wells and all(well in df.columns for well in blank_wells):
        avg_blank = df[blank_wells].mean(axis=1)
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=avg_blank,
            mode='lines',
            name='Average of All Blank Wells',
            line=dict(color='black', width=3)
        ))
    
    fig.update_layout(
        title='Blank Wells and Their Average',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white'
    )
    
    # Use a unique key to prevent reuse issues
    unique_key = f"plot_blank_wells_{uuid.uuid4().hex}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

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