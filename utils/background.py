import streamlit as st
import pandas as pd
import numpy as np

def perform_background_subtraction(groups_data, df, group_num, selected_blank_wells, selected_sample_wells):
    """
    Subtracts background (blank wells) from sample wells.
    """
    # Check if blank wells or sample wells are empty
    if not selected_blank_wells:
        st.warning(f"No blank wells selected for Group {group_num}. Cannot perform background subtraction.")
        return groups_data
        
    if not selected_sample_wells:
        st.warning(f"No sample wells selected for Group {group_num}. Cannot perform background subtraction.")
        return groups_data
        
    # Flatten any nested lists in selected_blank_wells
    flat_blank_wells = []
    for well in selected_blank_wells:
        if isinstance(well, list):
            flat_blank_wells.extend(well)
        else:
            flat_blank_wells.append(well)
            
    # Also flatten sample wells
    flat_sample_wells = []
    for well in selected_sample_wells:
        if isinstance(well, list):
            flat_sample_wells.extend(well)
        else:
            flat_sample_wells.append(well)
    
    # Make sure all blank wells exist in the dataframe
    missing_blank_cols = [bw for bw in flat_blank_wells if bw not in df.columns]
    if missing_blank_cols:
        st.error(f"These blank wells don't exist in the data: {missing_blank_cols}")
        return groups_data
        
    # Now use the flattened lists
    blank_mean = df[flat_blank_wells].mean(axis=1)
    
    # Create a new dataframe for background-subtracted data
    bg_subtracted = pd.DataFrame({'Time': df['Time']})
    
    # Process each sample well
    for well in flat_sample_wells:
        if well in df.columns:
            # Subtract background
            corrected_values = df[well] - blank_mean
            
            # Set negative values to zero
            corrected_values = corrected_values.clip(lower=0)
            
            # Add to the background-subtracted dataframe
            bg_subtracted[well] = corrected_values
    
    # Store the background-subtracted data in the groups_data dictionary
    groups_data[f"Group_{group_num}_bg_subtracted"] = bg_subtracted
    
    return groups_data
