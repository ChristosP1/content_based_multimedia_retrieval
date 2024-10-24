import streamlit as st
from prep_input_mesh import preprocess_input_mesh, find_similar_objects
import trimesh
import io

def search():
    '''UI for searching shapes and uploading files'''
    st.subheader("Upload 3D Shape File")
    uploaded_file = st.file_uploader(".", type=['obj'], label_visibility="collapsed")

    # Check if file is uploaded
    if uploaded_file is not None:
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        # If the file hasn't been processed yet, preprocess it
        if 'input_global_desc_df' not in st.session_state:
            try:
                # Read the uploaded file into trimesh
                mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), file_type='obj')
                
                # Call your preprocessing function with the mesh
                st.session_state['input_global_desc_df'], st.session_state['input_local_desc_df'] = preprocess_input_mesh(mesh)
            
            except Exception as e:
                st.error(f"Error loading mesh: {e}")
    
    col1, col2, col3 = st.columns([1, 2, 1]) 

    with col3:                       
        if uploaded_file and 'input_global_desc_df' in st.session_state:
            if st.button("Reprocess object"):
                try:
                    # Read the uploaded file into trimesh
                    mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), file_type='obj')
                        
                    # Call your preprocessing function with the mesh
                    st.session_state['input_global_desc_df'], st.session_state['input_local_desc_df'] = preprocess_input_mesh(mesh)
                    
                except Exception as e:
                    st.error(f"Error loading mesh: {e}")
        
    
    with col1:   
        # When the button is pressed, perform the similarity search
        if uploaded_file:
            if st.button("Find Similar Shapes", type="primary"):
                if 'input_global_desc_df' in st.session_state:
                    similar_meshes_global, similar_meshes_local, similar_meshes_total = find_similar_objects(st.session_state['input_global_desc_df'], st.session_state['input_local_desc_df'])
                    st.session_state['similar_meshes_global'] = similar_meshes_global
                    st.session_state['similar_meshes_local'] = similar_meshes_local
                    st.session_state['similar_meshes_total'] = similar_meshes_total
                else:
                    st.error("Please upload and preprocess the file first.")
                    
    col1_output, col2_output = st.columns([1, 1])
    
    if 'similar_meshes_global' in st.session_state and 'similar_meshes_local' in st.session_state:
        with col1_output:
            st.markdown("---")
            st.markdown("##### :small_orange_diamond: Similar objects global :small_orange_diamond:")
            st.write(st.session_state['similar_meshes_global'])
        with col2_output:
            st.markdown("---")
            st.markdown("##### :small_orange_diamond: Similar objects local :small_orange_diamond:")
            st.write(st.session_state['similar_meshes_local'])
    

    col1_output_total, col2_output_total, col3_output_total = st.columns([1, 8, 1])
    with col2_output_total:
        if 'similar_meshes_total' in st.session_state:
            st.markdown("---")
            st.markdown("##### :small_orange_diamond: Similar objects total :small_orange_diamond:")
            st.write(st.session_state['similar_meshes_total'])
        