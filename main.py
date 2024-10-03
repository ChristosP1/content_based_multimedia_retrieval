import streamlit as st
import ui
import ui_shape_viewer, ui_search_engine, ui_statistics
import os

ORIGINAL_SHAPES_PATH = 'datasets/dataset_snippet_small'
RESAMPLED_SHAPES_PATH = 'datasets/dataset_snippet_small_resample'


def main():
    '''
    The 'main' function creates the UI which includes various functionalities.
    Depending on the selected functionality, the corresponding code file is called.  
    '''
    
    # Set up the page title
    st.title("Content-Based Multimedia Retrieval")
    
    # Show sidebar navigation
    choice = ui.show_sidebar()

    # Render the selected page
    if choice == "Shape Viewer":
        if os.path.exists(ORIGINAL_SHAPES_PATH):
            ui_shape_viewer.shape_viewer(ORIGINAL_SHAPES_PATH, RESAMPLED_SHAPES_PATH)
        else:
            st.error("LabeledDB folder not found. Please check the path.")
    
    elif choice == "Similarity Search":
        uploaded_file = ui_search_engine.search()  # Capture the uploaded file from the UI

        # Process the uploaded file (if any)
        if uploaded_file is not None:
            # Here, you can add code to process or use the uploaded shape
            st.write(f"Processing file: {uploaded_file.name}")
    
    elif choice == "Global Statistics":
        ui_statistics.global_stats()
    
    elif choice == "Observations":
        st.write("Observations about the 3D objects will be added here :sunglasses:")

if __name__ == '__main__':
    main()
