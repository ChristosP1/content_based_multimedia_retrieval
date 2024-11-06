import streamlit as st
import plotly.graph_objects as go
import trimesh
import time


def update_progress(progress_callback, progress_value, delay=0.3):
    """
    Update the progress bar and yield control back to Streamlit.
    :param progress_callback: Function to update progress.
    :param progress_value: The progress value to set.
    :param delay: Time to sleep to simulate yielding control.
    """
    if progress_callback:
        progress_callback(progress_value)
        time.sleep(delay)  # Yield control to Streamlit to refresh UI


def load_object_file(filepath):
    '''
    Load OBJ file using Trimesh
    :param filepath: The path of the selected 3D object
    '''
    # Load the mesh from the OBJ file using Trimesh
    mesh = trimesh.load(filepath, file_type='obj')
    return mesh


def convert_mesh_to_plotly(mesh, rendering_mode, progress_callback=None):
    '''
    Convert Trimesh mesh to Plotly format
    :param mesh: The 3D object mesh that was previously loaded using Trimesh 
    :param rendering_mode: "Shaded" or "Wireframe"
    :param progress_callback: Function to update progress
    '''
    # Get vertices and triangles from the Trimesh mesh
    vertices = mesh.vertices
    triangles = mesh.faces

    # Simulate progress during the conversion process
    update_progress(progress_callback, 30)  # 30% done after extracting vertices and triangles

    traces = []  # Initialize empty list for traces

    if rendering_mode == "Shaded":
        # Create Plotly mesh trace for shaded mode
        trace = go.Mesh3d(
            x=vertices[:, 0],  # X-coordinates
            y=vertices[:, 1],  # Y-coordinates
            z=vertices[:, 2],  # Z-coordinates
            i=triangles[:, 0],  # First vertex index of triangle
            j=triangles[:, 1],  # Second vertex index of triangle
            k=triangles[:, 2],  # Third vertex index of triangle
            color='grey',  # Mesh color
            opacity=1,  # Set mesh transparency
            flatshading=True
        )
        traces.append(trace)  # Add shaded trace
    elif rendering_mode == "Wireframe":
        # Wireframe mode: add lines along edges of triangles
        edges = []
        for triangle in triangles:
            edges += [(triangle[i], triangle[j]) for i in range(3) for j in range(i + 1, 3)]

        edge_x = []
        edge_y = []
        edge_z = []

        for edge in edges:
            v0, v1 = vertices[edge[0]], vertices[edge[1]]
            edge_x += [v0[0], v1[0], None]  # X-coordinates of the edge
            edge_y += [v0[1], v1[1], None]  # Y-coordinates of the edge
            edge_z += [v0[2], v1[2], None]  # Z-coordinates of the edge

        # Add the wireframe trace as lines in the 3D plot
        wireframe_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=2),  # Black wireframe lines
            showlegend=False
        )
        traces.append(wireframe_trace)

    update_progress(progress_callback, 60)  # 60% done after creating the trace
    
    if rendering_mode == "Shaded + Edges":
        # Create Plotly mesh trace
        trace = go.Mesh3d(
            x=vertices[:, 0],  # X-coordinates
            y=vertices[:, 1],  # Y-coordinates
            z=vertices[:, 2],  # Z-coordinates
            i=triangles[:, 0],  # First vertex index of triangle
            j=triangles[:, 1],  # Second vertex index of triangle
            k=triangles[:, 2],  # Third vertex index of triangle
            color='#ffc72e',  # Mesh color
            opacity=0.5,  # Set mesh transparency
            flatshading=True
        )
    
        traces = [trace]  # Start with the surface trace

        update_progress(progress_callback, 60)  # 60% done after creating the trace
        
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

        update_progress(progress_callback, 100)  # 100% done after edges are processed
        
        

    return traces



def visualize_3d_shape(filepath, rendering_mode, progress_callback=None, width=700, height=700, displayModeBar=True):
    '''
    Render the 3D mesh using Plotly in Streamlit
    :param filepath: The path of the selected 3D object
    :param show_edges: Boolean to indicate whether to show mesh edges
    :param rendering_mode: Either 'Shaded' or 'Wireframe'
    '''
    # Initialize a progress bar
    progress_bar = st.progress(0)

    # Load the mesh file
    mesh = load_object_file(filepath)
    
    update_progress(progress_callback, 10)  # 10% done after loading the mesh

    # Convert the mesh to Plotly format with a progress callback
    traces = convert_mesh_to_plotly(mesh, rendering_mode, progress_callback=lambda progress: progress_bar.progress(progress))

    # Define layout for the border (rectangle around the chart)
    layout = go.Layout(
        width=width,  # Set the width of the plot
        height=height,  # Set the height of the plot
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
                line=dict(color="black", width=1)  # Border color and width
            )
        ]
    )

    # Create Plotly figure with the trace and layout (with border)
    fig = go.Figure(data=traces, layout=layout)
    
    # Render the figure in Streamlit 
    st.plotly_chart(fig, config={'displayModeBar': displayModeBar})

    # # Clear the progress bar once the work is done
    progress_bar.empty()

