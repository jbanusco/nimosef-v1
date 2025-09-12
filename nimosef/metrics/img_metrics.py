import numpy as np
import pandas as pd
import numpy as np

from skimage.measure import label


def count_connected_components(segmentation, background=0):
    """
    Count the number of connected components for each label in a segmentation image.
    
    Args:
        segmentation (np.array): 2D or 3D array with integer labels.
        background (int): Label for background (components with this label are ignored).
        
    Returns:
        dict: Mapping from label to number of connected components.
    """
    unique_labels = np.unique(segmentation)
    comp_counts = {}
    for lbl in unique_labels:
        if lbl == background:
            continue
        # Create binary mask for the label.
        mask = segmentation == lbl
        # Label connected components.
        labeled_components = label(mask)
        comp_counts[int(lbl)] = int(labeled_components.max())  # number of components
    return comp_counts


def count_connected_components_3d_df(segmentation_4d, background=0, time_axis=0):
    """
    Compute the number of connected components for each label in each 3D volume (time frame)
    of a 4D segmentation array, and return the results in a DataFrame.
    
    Args:
        segmentation_4d (np.ndarray): 4D segmentation array.
        background (int): Label to ignore.
        time_axis (int): The axis representing time.
        
    Returns:
        pd.DataFrame: DataFrame with columns: "time", "label", "component_count".
    """
    # Move the time axis to the front if it isn't already.
    if time_axis != 0:
        segmentation_4d = np.moveaxis(segmentation_4d, time_axis, 0)
    num_frames = segmentation_4d.shape[0]
    records = []
    for t in range(num_frames):
        volume = segmentation_4d[t]  # This is a 3D volume.
        unique_labels = np.unique(volume)
        for lbl in unique_labels:
            if lbl == background:
                continue
            mask = (volume == lbl)
            labeled = label(mask)
            comp_count = int(labeled.max())  # number of connected components
            records.append({"time": t, "label": lbl, "component_count": comp_count})
    return pd.DataFrame(records)


def summarize_component_counts(df_components):
    """
    For a DataFrame of connected component counts per time frame,
    group by label and return a summary with both the mean and maximum 
    component counts.
    
    Args:
        df_components (pd.DataFrame): DataFrame with columns "time", "label", and "component_count".
    
    Returns:
        pd.DataFrame: A summary DataFrame with columns "label", "avg_components", and "max_components".
    """
    summary = (
        df_components
        .groupby("label")["component_count"]
        .agg(avg_components="mean", max_components="max")
        .reset_index()
    )
    return summary