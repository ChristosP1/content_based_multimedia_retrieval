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
    mesh = o3d.io.read_triangle_mesh(filepath, enable_post_processing=True)
    return mesh


def convert_mesh_to_plotly(mesh, show_edges):
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
        flatshading=True
    )
    
    traces = [trace]  # Start with the surface trace
    
    if show_edges:
        # Add edges as lines between the vertices
        edges = []
        for triangle in triangles:
            edges += [(triangle[i], triangle[j]) for i in range(3) for j in range(i+1, 3)]

        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in edges:
            v0, v1 = vertices[edge[0]], vertices[edge[1]]
            edge_x += [v0[0], v1[0], None]  # X-coordinates of the edge
            edge_y += [v0[1], v1[1], None]  # Y-coordinates of the edge
            edge_z += [v0[2], v1[2], None]  # Z-coordinates of the edge

        # Add the edge trace as lines in the 3D plot
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=2),  # Black edges
            showlegend=False
        )
        traces.append(edge_trace)  # Add edges trace
    
    return traces


def visualize_3d_shape(filepath, show_edges):
    '''
    Render the 3D mesh using Plotly in Streamlit
    :param filepath: The path of the selected 3D object
    :param show_edges: Boolean to indicate whether to show mesh edges
    '''
    mesh = load_off_file(filepath)  # Load the OFF file
    traces = convert_mesh_to_plotly(mesh, show_edges)  # Convert mesh to Plotly format

    # Define layout for the border (rectangle around the chart)
    layout = go.Layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=True, zeroline=False, showspikes=False),
            yaxis=dict(showbackground=False, showgrid=True, zeroline=False, showspikes=False),
            zaxis=dict(backgroundcolor="#f3f4f5", showbackground=True, showgrid=True, zeroline=False, showspikes=False),
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
    fig = go.Figure(data=traces, layout=layout)

    # Render the figure in Streamlit
    st.plotly_chart(fig)
