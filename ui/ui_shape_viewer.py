import streamlit as st
import os
from ui.ui_visualize3D import visualize_3d_shape

def shape_viewer(original_db_path, resampled_db_path):
    '''
    UI for viewing 3D shapes
    :param original_db_path: The path of the root directory of the original shapes
    :param resampled_db_path: The path of the root directory of the resampled shapes
    '''
    st.subheader("Select and view objects")
    
    # Choose if you want to view the original or the resampled database
    show_resampled = st.toggle("Final objects")
    db_path = resampled_db_path if show_resampled else original_db_path

     
    # Get all subdirectories (shape categories) in the dataset folder
    categories = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d))]
    selected_category = st.selectbox("Choose a category", categories)
    
    # Display all OFF files in the selected category
    shape_files = [f for f in os.listdir(os.path.join(db_path, selected_category)) if f.endswith('.obj')]
    selected_shape = st.selectbox("Choose a shape file", shape_files)
    
    # Display selected shape info
    # st.text(f"Displaying shape: '{selected_shape}'   |   Category: '{selected_category}'")
    # st.text(f"Category: {selected_category}")

    # Get full path to the selected shape
    shape_path = os.path.join(db_path, selected_category, selected_shape)
    
    # Dropdown to select rendering mode
    rendering_mode = st.radio("Rendering Mode", options=["Shaded", "Shaded + Edges", "Wireframe"], horizontal=True)
    
    # Button to visualize the shape
    if st.button("Visualize Shape", type="primary"):
        visualize_3d_shape(shape_path, rendering_mode)