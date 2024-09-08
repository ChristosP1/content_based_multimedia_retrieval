import streamlit as st
import ui
import os

LABELED_DB_PATH = "original_shapes"


def main():
    '''
    The 'main' function creates the UI which includes various functionalities.
    Depending on the selected functionality, the corresponding code file is called.  
    '''
    
    # Set up the page title
    st.title("Content-Dased Multimedia Retrieval")
    
    # Show sidebar navigation
    choice = ui.show_sidebar()

    # Render the selected page
    if choice == "Shape Viewer":
        if os.path.exists(LABELED_DB_PATH):
            ui.shape_viewer(LABELED_DB_PATH)
        else:
            st.error("LabeledDB folder not found. Please check the path.")
    
    elif choice == "Similarity Search":
        uploaded_file = ui.search()  # Capture the uploaded file from the UI

        # Process the uploaded file (if any)
        if uploaded_file is not None:
            # Here, you can add code to process or use the uploaded shape
            st.write(f"Processing file: {uploaded_file.name}")
    
    elif choice == "Statistics":
        st.write("Statistics about the 3D objects will be added here")

if __name__ == '__main__':
    main()
