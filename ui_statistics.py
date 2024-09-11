import pandas as pd
import streamlit as st

def global_stats():
    # Load global statistics from CSV
    global_stats = pd.read_csv(r'outputs/data/global_statistics.csv')

    # Extract the row with statistics (assuming it's the first row)
    stats_row = global_stats.iloc[0]

    # Define a dictionary of statistics and their corresponding names
    stats = {
        "Total Shapes": stats_row['total_shapes'],
        "Mean Vertices": stats_row['mean_vertices'],
        "Std Vertices": stats_row['std_vertices'],
        "Mean Faces": stats_row['mean_faces'],
        "Std Faces": stats_row['std_faces'],
        "Has Holes Count": stats_row['has_holes_count'],
        "Outlier Low Count": stats_row['outlier_low_count'],
        "Outlier High Count": stats_row['outlier_high_count']
    }

    # Display the statistics in a grid with Streamlit's layout
    st.title("Global Statistics")

    # Use columns to show metrics in a compact layout
    col1, col2, col3 = st.columns(3)

    # First Column
    with col1:
        st.metric(label="Total Shapes", value=int(stats["Total Shapes"]))
        st.metric(label="Std Vertices", value=f"{stats['Std Vertices']:.2f}")
        st.metric(label="Has Holes Count", value=int(stats["Has Holes Count"]))
    # Second Column
    with col2:
        st.metric(label="Mean Faces", value=f"{stats['Mean Faces']:.2f}")
        st.metric(label="Std Faces", value=f"{stats['Std Faces']:.2f}")
        st.metric(label="Outlier Low Count", value=int(stats["Outlier Low Count"]))
    # Third Column
    with col3:
        st.metric(label="Mean Vertices", value=f"{stats['Mean Vertices']:.2f}")
        
        
        st.metric(label="Outlier High Count", value=int(stats["Outlier High Count"]))
