import streamlit as st

def show_sidebar():

    '''Function to create sidebar navigation'''
    st.sidebar.title("Options")
    return st.sidebar.radio(".", ["Shape Viewer", "Search Engine", "Presentation"], label_visibility="collapsed")


