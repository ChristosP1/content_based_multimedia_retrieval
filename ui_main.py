import streamlit as st
import ui.ui as ui
import ui.ui_shape_viewer as ui_shape_viewer
import ui.ui_search_engine as ui_search_engine
import ui.ui_statistics as ui_statistics
import ui.ui_presentation as ui_presentation
import os

ORIGINAL_SHAPES_PATH = 'datasets/dataset_snippet_medium'
RESAMPLED_SHAPES_PATH = 'datasets/dataset_snippet_medium_normalized'


def main():
    '''
    The 'main' function creates the UI which includes various functionalities.
    Depending on the selected functionality, the corresponding code file is called.  
    '''
    
    
    # Set up the page title
    
    
    # Show sidebar navigation
    choice = ui.show_sidebar()

    # Render the selected page
    if choice == "Shape Viewer":
        st.title("Shape Viewer")
        if os.path.exists(ORIGINAL_SHAPES_PATH):
            ui_shape_viewer.shape_viewer(ORIGINAL_SHAPES_PATH, RESAMPLED_SHAPES_PATH)
        else:
            st.error("LabeledDB folder not found. Please check the path.")
    
    elif choice == "Search Engine":
        uploaded_file = ui_search_engine.search()  # Capture the uploaded file from the UI
        # Process the uploaded file (if any)
        if uploaded_file is not None:
            # Here, you can add code to process or use the uploaded shape
            st.write(f"Processing file: {uploaded_file.name}")
    
    # elif choice == "Global Statistics":
    #     ui_statistics.global_stats()
    
    elif choice == "Presentation":
        ui_presentation.presentation()

if __name__ == '__main__':
    main()
