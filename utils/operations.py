import pandas as pd
import numpy as np

def perform_group_operations(group1_data, group2_data, operation):
    """
    Lines ~1280-~1340 in your script: 
    Add/Sub/Mul/Div group1_data and group2_data with dynamic labeling.
    """
    operated_data=pd.DataFrame({"Time":group1_data["Time"]})
    sample_wells1=list(group1_data.columns[1:])
    sample_wells2=list(group2_data.columns[1:])
    min_len=min(len(sample_wells1), len(sample_wells2))
    sample_wells1=sample_wells1[:min_len]
    sample_wells2=sample_wells2[:min_len]
    symbol_map={"Add":"+","Subtract":"-","Multiply":"*","Divide":"/"}
    symbol=symbol_map.get(operation,"+")

    for w1,w2 in zip(sample_wells1, sample_wells2):
        new_label=f"{w1} {symbol} {w2}"
        if operation=="Add":
            operated_data[new_label]=group1_data[w1]+group2_data[w2]
        elif operation=="Subtract":
            operated_data[new_label]=group1_data[w1]-group2_data[w2]
        elif operation=="Multiply":
            operated_data[new_label]=group1_data[w1]*group2_data[w2]
        elif operation=="Divide":
            operated_data[new_label]=group1_data[w1]/group2_data[w2].replace(0,np.nan)

    if operated_data.shape[1]>1:
        operated_data["Average"]=operated_data.iloc[:,1:].mean(axis=1)
    return operated_data
