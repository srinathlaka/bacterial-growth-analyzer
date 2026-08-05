import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter

def detect_phases(
    time,
    od_values,
    method="thresholded_derivative",
    smoothing_window=5,
    polyorder=2,
    derivative_threshold=0.005,
    min_distance=5,
    slope_window=5,
    slope_threshold=0.001
):
    """
    Lines 1-~120 from your code: the detect_phases function. 
    Also handles slope_based. 
    Returns phases, breakpoints, extra.
    """
    phases=[]
    breakpoints=[]
    extra={}

    if method=="thresholded_derivative":
        if smoothing_window%2==0:
            smoothing_window+=1
        try:
            smoothed=savgol_filter(od_values, window_length=smoothing_window, polyorder=polyorder)
        except ValueError:
            st.error("Savgol filter error: window_length might be too large for the data length.")
            return [],[],{}
        derivative=np.gradient(smoothed,time)
        extra["derivative"]=derivative

        sign=np.sign(derivative)
        sign_changes=[]
        for i in range(1,len(sign)):
            if sign[i]!=sign[i-1]:
                if abs(derivative[i])>derivative_threshold or abs(derivative[i-1])>derivative_threshold:
                    sign_changes.append(i)
        filtered_changes=[]
        last_cp=-min_distance
        for cp in sign_changes:
            if cp-last_cp>=min_distance:
                filtered_changes.append(cp)
                last_cp=cp
        breakpoints=filtered_changes

        start_idx=0
        for bp in breakpoints:
            end_idx=bp
            if end_idx==0:
                continue
            phases.append((time[start_idx], time[end_idx-1]))
            start_idx=bp
        if start_idx<len(time):
            phases.append((time[start_idx], time[-1]))

    elif method=="slope_based":
        if len(time)<=slope_window:
            st.error(
                f"Not enough data points ({len(time)}) for a slope window of {slope_window}. "
                f"Reduce the slope window or use more data."
            )
            return [],[],{}
        slopes=[]
        for i in range(len(time)-slope_window):
            y_segment=od_values[i:i+slope_window]
            x_segment=time[i:i+slope_window]
            A=np.vstack([x_segment, np.ones(len(x_segment))]).T
            m,_=np.linalg.lstsq(A, y_segment, rcond=None)[0]
            slopes.append(m)
        slopes=np.array(slopes+[slopes[-1]]*slope_window)
        extra["slopes"]=slopes

        high_slope=np.abs(slopes)>slope_threshold
        slope_changes=[]
        for i in range(1,len(high_slope)):
            if high_slope[i]!=high_slope[i-1]:
                slope_changes.append(i)
        breakpoints=slope_changes

        start_idx=0
        for bp in breakpoints:
            phases.append((time[start_idx], time[bp-1]))
            start_idx=bp
        if start_idx<len(time):
            phases.append((time[start_idx], time[-1]))
    else:
        st.error("Unknown method. Use 'thresholded_derivative' or 'slope_based'.")
        return [],[],{}

    return phases, breakpoints, extra

def plot_detected_phases(time, od_values, phases, derivative=None, slopes=None, change_points=None):
    """
    Plot the OD values with detected phases highlighted, plus derivative/slopes if provided.
    """
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=od_values,
        mode='lines+markers',
        name='OD Values',
        line=dict(color='blue')
    ))
    if derivative is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=derivative,
            mode='lines',
            name='Derivative',
            line=dict(color='orange', dash='dash')
        ))
    if slopes is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=slopes,
            mode='lines',
            name='Slopes',
            line=dict(color='purple', dash='dot')
        ))
    colors=[
        'rgba(255,0,0,0.2)',
        'rgba(0,255,0,0.2)',
        'rgba(0,0,255,0.2)',
        'rgba(255,255,0,0.2)',
        'rgba(255,165,0,0.2)'
    ]
    for idx,(start,end) in enumerate(phases):
        color=colors[idx%len(colors)]
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0,
            annotation_text=f"Phase {idx+1}",
            annotation_position="top left"
        )
    if change_points is not None:
        for cp in change_points:
            if cp<len(time):
                fig.add_vline(
                    x=time[cp],
                    line=dict(color='red', dash='dot'),
                    annotation_text="Change Point",
                    annotation_position="top left"
                )
    fig.update_layout(
        title="Automatic Phase Detection",
        xaxis_title="Time",
        yaxis_title="OD",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
