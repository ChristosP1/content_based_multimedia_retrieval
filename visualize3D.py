import streamlit as st
import open3d as o3d
import plotly.graph_objects as go
import numpy as np


def load_off_file(filepath):
    '''
    Load OFF file using Open3D
    :param filepath: The path of the selected 3D object
    '''
    # Load the mesh from the OFF file
    mesh = o3d.io.read_triangle_mesh(filepath)
    return mesh


def convert_mesh_to_plotly(mesh):
    '''
    Convert Open3D mesh to Plotly format
    :param mesh: The 3D object mesh that was previously loaded using Open3D 
    '''
    # Get vertices and triangles from the Open3D mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create Plotly mesh trace
    trace = go.Mesh3d(
        x=vertices[:, 0],  # X-coordinates
        y=vertices[:, 1],  # Y-coordinates
        z=vertices[:, 2],  # Z-coordinates
        i=triangles[:, 0],  # First vertex index of triangle
        j=triangles[:, 1],  # Second vertex index of triangle
        k=triangles[:, 2],  # Third vertex index of triangle
        color='lightblue',  # Mesh color
        opacity=0.5,  # Set mesh transparency
    )
    return trace


def visualize_3d_shape(filepath):
    '''
    Render the 3D mesh using Plotly in Streamlit
    :param filepath: The path of the selected 3D object
    '''
    mesh = load_off_file(filepath)  # Load the OFF file
    trace = convert_mesh_to_plotly(mesh)  # Convert mesh to Plotly format

    # Define layout for the border (rectangle around the chart)
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                showbackground=False,  
                showgrid=True,       
                zeroline=False,        
                showspikes=False      
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=True,
                zeroline=False,
                showspikes=False
            ),
            zaxis=dict(
                backgroundcolor="#f3f4f5",
                showbackground=True,
                showgrid=True,
                zeroline=False,
                showspikes=False
            ),
            aspectmode='data'
        ),
        # Add a rectangular shape (border) around the 3D chart area
        shapes=[
            dict(
                type="rect",
                x0=0, y0=0, x1=1, y1=1,
                xref="paper", yref="paper",
                line=dict(color="lightblue", width=1)  # Border color and width
            )
        ]
    )

    # Create Plotly figure with the trace and layout (with border)
    fig = go.Figure(data=[trace], layout=layout)

    # Render the figure in Streamlit
    st.plotly_chart(fig)