import pandas as pd
import streamlit as st

def global_stats():
    # Load global statistics from CSV
    global_stats_original = pd.read_csv(r'outputs/data/global_statistics_original.csv')
    global_stats_final = pd.read_csv(r'outputs/data/global_statistics_final.csv')


    # -------------------------------------------------------------------------------------------------------------------------
    # Extract the row with statistics (assuming it's the first row)
    original_stats_row = global_stats_original.iloc[0]

    # Define a dictionary of statistics and their corresponding names
    original_stats = {
        "Total Shapes": original_stats_row['total_shapes'],
        "Mean Vertices": original_stats_row['mean_vertices'],
        "Std Vertices": original_stats_row['std_vertices'],
        "Mean Faces": original_stats_row['mean_faces'],
        "Std Faces": original_stats_row['std_faces'],
        "Manifold Count": original_stats_row['is_manifold'],
        "Outlier Low Count": original_stats_row['outlier_low_count'],
        "Outlier High Count": original_stats_row['outlier_high_count']
    }

    # Display the statistics in a grid with Streamlit's layout
    st.title("Global Statistics Before")

    # Use columns to show metrics in a compact layout
    col1, col2, col3 = st.columns(3)

    # First Column
    with col1:
        st.metric(label="Total Shapes", value=int(original_stats["Total Shapes"]))
        st.metric(label="Std Vertices", value=f"{original_stats['Std Vertices']:.2f}")
        st.metric(label="Manifold Count", value=int(original_stats["Manifold Count"]))
    # Second Column
    with col2:
        st.metric(label="Mean Faces", value=f"{original_stats['Mean Faces']:.2f}")
        st.metric(label="Std Faces", value=f"{original_stats['Std Faces']:.2f}")
        st.metric(label="Outlier Low Count", value=int(original_stats["Outlier Low Count"]))
    # Third Column
    with col3:
        st.metric(label="Mean Vertices", value=f"{original_stats['Mean Vertices']:.2f}")
        
        
        st.metric(label="Outlier High Count", value=int(original_stats["Outlier High Count"]))
        
    
    
    # -------------------------------------------------------------------------------------------------------------------------
    # Extract the row with statistics (assuming it's the first row)
    final_stats_row = global_stats_final.iloc[0]

    # Define a dictionary of statistics and their corresponding names
    final_stats = {
        "Total Shapes": final_stats_row['total_shapes'],
        "Mean Vertices": final_stats_row['mean_vertices'],
        "Std Vertices": final_stats_row['std_vertices'],
        "Mean Faces": final_stats_row['mean_faces'],
        "Std Faces": final_stats_row['std_faces'],
        "Manifold Count": final_stats_row['is_manifold'],
        "Outlier Low Count": final_stats_row['outlier_low_count'],
        "Outlier High Count": final_stats_row['outlier_high_count']
    }

    # Display the statistics in a grid with Streamlit's layout
    st.title("Global Statistics After")

    # Use columns to show metrics in a compact layout
    col1, col2, col3 = st.columns(3)

    # First Column
    with col1:
        st.metric(label="Total Shapes", value=int(final_stats["Total Shapes"]))
        st.metric(label="Std Vertices", value=f"{final_stats['Std Vertices']:.2f}")
        st.metric(label="Manifold Count", value=int(final_stats["Manifold Count"]))
    # Second Column
    with col2:
        st.metric(label="Mean Faces", value=f"{final_stats['Mean Faces']:.2f}")
        st.metric(label="Std Faces", value=f"{final_stats['Std Faces']:.2f}")
        st.metric(label="Outlier Low Count", value=int(final_stats["Outlier Low Count"]))
    # Third Column
    with col3:
        st.metric(label="Mean Vertices", value=f"{final_stats['Mean Vertices']:.2f}")
        
        
        st.metric(label="Outlier High Count", value=int(final_stats["Outlier High Count"]))
        
        
        
        

    
    
    
    
