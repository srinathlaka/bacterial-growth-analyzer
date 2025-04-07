import streamlit as st
import pandas as pd
import numpy as np

def perform_background_subtraction(groups_data, df, group_num, selected_blank_wells, selected_sample_wells):
    """
    From lines ~400 or so. 
    Subtract average blank from sample wells, store in groups_data.
    """
    if len(selected_sample_wells)==0:
        st.warning(f"No sample wells selected for Group {group_num}.")
        return groups_data
    if len(selected_blank_wells)==0:
        st.warning(f"No blank wells selected for Group {group_num}. Subtracting zero.")
        blank_mean=0
    else:
        blank_mean=df[selected_blank_wells].mean(axis=1)

    group_df=df.copy()
    for sample_well in selected_sample_wells:
        group_df[sample_well]=(df[sample_well]-blank_mean).clip(lower=0)

    groups_data[f"Group_{group_num}_bg_subtracted"]=group_df[["Time"]+selected_sample_wells]
    groups_data[f"Group_{group_num}_sample_wells"]=selected_sample_wells
    return groups_data
