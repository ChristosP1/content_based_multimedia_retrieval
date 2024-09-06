import streamlit as st
import ui

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
        ui.shape_viewer()
    
    elif choice == "Similarity Search":
        uploaded_file = ui.search()  # Capture the uploaded file from the UI

        # Process the uploaded file (if any)
        if uploaded_file is not None:
            # Here, you can add code to process or use the uploaded shape
            st.write(f"Processing file: {uploaded_file.name}")

if __name__ == '__main__':
    main()
