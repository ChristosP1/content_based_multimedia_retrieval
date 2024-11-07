import streamlit as st
from prep_input_mesh import (preprocess_input_mesh, compute_distances, 
                             standardize_and_save_similarity_scores, 
                             find_dominant_class, get_dominant_class_items)
from ui.ui_visualize3D import visualize_3d_shape
import trimesh
import io
from PIL import Image
import matplotlib.pyplot as plt
import os


def display_3d_views_for_top_objects(df, rendering_mode="Shaded", top_objects_num=False):
    """
    Displays 3D viewers for the top similar objects in a 2-column grid layout.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'file_name' and 'obj_class'.
        col3_output (streamlit DeltaGenerator): Column in Streamlit layout to display the 3D views.
        rendering_mode (str): Rendering mode for 3D visualization ("Shaded", "Wireframe", etc.)
    """
    # Limit to top 10 objects for display
    if top_objects_num:
        top_objects = df.head(top_objects_num)
        
        # Split objects into rows with 2 items each for a 2-column layout
        for i in range(0, len(top_objects), 3):
            row = top_objects.iloc[i:i+3]  # Select two objects per row

            # Create a row with two columns in col3_output
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if j < len(row):
                    obj_data = row.iloc[j]
                    file_name = obj_data['file_name']
                    obj_class = obj_data['obj_class']
                        
                    # Construct the path for each .obj file
                    obj_path = f"datasets/dataset_snippet_medium_normalized/{obj_class}/{file_name}"

                    # Display the file name and class
                    col.write(f"Object: {file_name} (Class: {obj_class})")
                        
                    # Render 3D object
                    if os.path.exists(obj_path):
                        with col:
                                visualize_3d_shape(obj_path, rendering_mode, width=200, height=350, displayModeBar=False)
                    else:
                        col.warning(f"File not found: {obj_path}")
                        
    else:
        # Split objects into rows with 2 items each for a 2-column layout
        for i in range(0, len(df), 3):
            row = df.iloc[i:i + 3]  # Select up to 3 objects for each row

            # Create a new row with 3 columns in Streamlit
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if j < len(row):
                    obj_data = row.iloc[j]
                    file_name = obj_data['file_name']
                    obj_class = obj_data['obj_class']
                        
                    # Construct the path for each .obj file
                    obj_path = f"datasets/dataset_snippet_medium_normalized/{obj_class}/{file_name}"

                    # Display the file name and class
                    col.write(f"Object: {file_name} (Class: {obj_class})")
                        
                    # Render 3D object
                    if os.path.exists(obj_path):
                        with col:
                            visualize_3d_shape(obj_path, rendering_mode, width=200, height=350, displayModeBar=False)
                    else:
                        col.warning(f"File not found: {obj_path}")
    

def search():
    '''UI for searching shapes and uploading files'''
    st.subheader("Upload 3D Shape File")
    uploaded_file = st.file_uploader(".", type=['obj'], label_visibility="collapsed")
    
    # Initialize session state for descriptors and results if not already present
    if 'input_desc_df' not in st.session_state:
        st.session_state['input_desc_df'] = None
    if 'similar_meshes' not in st.session_state:
        st.session_state['similar_meshes'] = None
    enhanced_search = False

    # Check if file is uploaded
    if uploaded_file is not None:
        # st.write(f"Uploaded file: {uploaded_file.name}")
        
        if st.session_state['input_desc_df'] is None:
            try:
                # st.write("Start preprocessing")
                # Read the uploaded file into trimesh
                mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), file_type='obj')
                
                # Process the mesh and save descriptors in session state
                st.session_state['input_desc_df'] = preprocess_input_mesh(mesh)
            
            except Exception as e:
                st.error(f"Error loading mesh: {e}")
    
    # Display the processed descriptors
    # if st.session_state['input_desc_df'] is not None:
    #     st.write("Processed descriptors:", st.session_state['input_desc_df'])

    col1, col2, col3 = st.columns([1, 2, 1])  

    with col3:                       
        if uploaded_file and st.session_state['input_desc_df'] is not None:
            if st.button("Reprocess object"):
                try:
                    st.write("Reprocessing object")
                    # Read and reprocess the uploaded file
                    mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), file_type='obj')
                    st.session_state['input_desc_df'] = preprocess_input_mesh(mesh)
                    # Clear any previous search results
                    st.session_state['similar_meshes'] = None
                    
                except Exception as e:
                    st.error(f"Error loading mesh: {e}")
        
    with col1:   
        if uploaded_file:
            enhanced_search = st.toggle("Enhanced search")
            if st.button("Find Similar Shapes", type="primary"):
                if st.session_state['input_desc_df'] is not None:
                    # Compute distances and save results in session state
                    distances = compute_distances(st.session_state['input_desc_df'])
                    similar_meshes = standardize_and_save_similarity_scores(distances)
                    st.session_state['similar_meshes'] = similar_meshes
                    
                    if enhanced_search:
                        dominant_class = find_dominant_class(similar_meshes)
                        st.write(f"Dominant class: {dominant_class}")
                        st.session_state['dominant_class'] = dominant_class
                        dominant_class_items = get_dominant_class_items(similar_meshes, dominant_class)
                        st.session_state['dominant_class_items'] = dominant_class_items
                        
                else:
                    st.error("Please upload and preprocess the file first.")
                    
    
    
    # Display search results if available
    col1_output, col2_output, col3_output = st.columns([10, 1, 2])
    if st.session_state['similar_meshes'] is not None and uploaded_file:
        st.markdown("---")
        if not enhanced_search or not dominant_class:
            with col1_output:
                st.markdown("##### :small_orange_diamond: Similar objects :small_orange_diamond:")
                st.write(st.session_state['similar_meshes'][['file_name', 'obj_class', 'final_score']].head(10))
                
            display_3d_views_for_top_objects(st.session_state['similar_meshes'], rendering_mode="Shaded + Edges", top_objects_num=6)
        else:
            with col1_output:
                st.markdown("##### :small_orange_diamond: Similar objects :small_orange_diamond:")
                st.write(st.session_state['dominant_class_items'][['file_name', 'obj_class', 'final_score']].head(10))
            
            display_3d_views_for_top_objects(st.session_state['dominant_class_items'], rendering_mode="Shaded + Edges")
       
        