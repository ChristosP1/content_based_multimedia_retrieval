import streamlit as st
import ui.components as ui

def main():
    # Set up the page title
    st.title("3D Shape Retrieval System")
    
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
