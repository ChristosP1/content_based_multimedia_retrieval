import streamlit as st
import os
from visualize3D import visualize_3d_shape

def show_sidebar():
    '''Function to create sidebar navigation'''
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Shape Viewer", "Similarity Search", "Statistics"])

def shape_viewer(labeled_db_path):
    '''
    UI for viewing 3D shapes
    :param labeled_db_path: The path of the root directory that contains the OFF files of the 3D objects
    '''
    st.subheader("3D Shape Viewer")
    
    # Get all subdirectories (shape categories) in the dataset folder
    categories = [d for d in os.listdir(labeled_db_path) if os.path.isdir(os.path.join(labeled_db_path, d))]
    selected_category = st.selectbox("Choose a category", categories)
    
    # Display all OFF files in the selected category
    shape_files = [f for f in os.listdir(os.path.join(labeled_db_path, selected_category)) if f.endswith('.obj')]
    selected_shape = st.selectbox("Choose a shape file", shape_files)
    
    # Display selected shape info
    st.text(f"Displaying shape: {selected_shape}")
    st.text(f"Category: {selected_category}")

    # Get full path to the selected shape
    shape_path = os.path.join(labeled_db_path, selected_category, selected_shape)

    # Checkbox to toggle mesh edges on and off
    show_edges = st.checkbox("Show Mesh Edges", value=False)
    
    # Button to visualize the shape
    if st.button("Visualize Shape"):
        visualize_3d_shape(shape_path, show_edges)
    
    

def search():
    '''UI for searching shapes and uploading files'''
    st.subheader("Upload 3D Shape File")
    uploaded_file = st.file_uploader("Choose a 3D shape file", type=['obj', 'ply', 'off'])
    
    if uploaded_file is not None:
        st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Placeholder for future search button
    if st.button("Find Similar Shapes"):
        st.text("Functionality for similarity search will be implemented here.")
    
    return uploaded_file  # Return the uploaded file so it can be used in main
