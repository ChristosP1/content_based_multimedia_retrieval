import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from trimesh import load as load_object_file
import numpy as np
import os

def load_data(csv_file):
    return pd.read_csv(csv_file)

def plot_histograms(df):
    st.markdown("### Distribution of Vertices and Faces")
    fig_vertices = px.histogram(df, x='vertices', height=350,
                                title="Distribution of Vertices", color_discrete_sequence=['black'])
    
    fig_faces = px.histogram(df, x='faces', height=350, 
                             title="Distribution of Faces", color_discrete_sequence=['red'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_vertices, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_faces, use_container_width=True)


def show_global_statistics(df):
    st.markdown("### Global statistics")
    stats_row = df.iloc[0]
    stats = {
        "Total Shapes": stats_row['total_shapes'],
        "Mean Vertices": stats_row['mean_vertices'],
        "Std Vertices": stats_row['std_vertices'],
        "Mean Faces": stats_row['mean_faces'],
        "Std Faces": stats_row['std_faces'],
        "Watertight Count": stats_row['is_watertight'],
        "Outlier Low Count": stats_row['outlier_low_count'],
        "Outlier High Count": stats_row['outlier_high_count']
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Shapes", value=int(stats["Total Shapes"]))
        st.metric(label="Mean Vertices", value=f"{stats['Mean Vertices']:.2f}")
    with col2:
        st.metric(label="Watertight Count", value=int(stats["Watertight Count"]))
        st.metric(label="Mean Faces", value=f"{stats['Mean Faces']:.2f}")
    with col3:
        st.metric(label="Outlier Low Count", value=int(stats["Outlier Low Count"]))
        st.metric(label="Std Vertices", value=f"{stats['Std Vertices']:.2f}")
    with col4:
        st.metric(label="Outlier High Count", value=int(stats["Outlier High Count"]))
        st.metric(label="Std Faces", value=f"{stats['Std Faces']:.2f}")


def convert_mesh_to_plotly(mesh):
    vertices = mesh.vertices
    triangles = mesh.faces

    traces = []  
    
    trace = go.Mesh3d(
        x=vertices[:, 0],  
        y=vertices[:, 1],  
        z=vertices[:, 2],  
        i=triangles[:, 0],  
        j=triangles[:, 1],  
        k=triangles[:, 2],  
        color='#ffc72e',  
        opacity=1,  
        flatshading=True)
    
    traces = [trace]  
        
    edges = []
    for triangle in triangles:
        edges += [(triangle[i], triangle[j]) for i in range(3) for j in range(i+1, 3)]

    edge_x = []
    edge_y = []
    edge_z = []
        
    for edge in edges:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        edge_x += [v0[0], v1[0], None] 
        edge_y += [v0[1], v1[1], None] 
        edge_z += [v0[2], v1[2], None] 
            
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=2),  
        showlegend=False
    )
    traces.append(edge_trace)  

    return traces


def plot_3d_shape_interface(file_path):
   
    if file_path is not None:        
        mesh = load_object_file(file_path)

        traces = convert_mesh_to_plotly(mesh)

        layout = go.Layout(
            width=800,
            height=500,
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=True, zeroline=False, showspikes=False),
                yaxis=dict(showbackground=False, showgrid=True, zeroline=False, showspikes=False),
                zaxis=dict(showbackground=False, showgrid=True, zeroline=False, showspikes=False),
                aspectmode='data' 
            ),
            
            shapes=[
            dict(
                type="rect",
                x0=0, y0=0, x1=1, y1=1,
                xref="paper", yref="paper",
                line=dict(color="black", width=1)
            )
        ]
        )

        fig = go.Figure(data=traces, layout=layout)

        col1, col2, col3 = st.columns([1, 10, 1]) 

        with col2:
            st.plotly_chart(fig)


def convert_mesh_to_voxels(mesh, pitch=0.1):
    """
    Convert a 3D mesh into a voxel grid.
    :param mesh: A Trimesh object representing the 3D mesh
    :param pitch: The voxel size (distance between voxel centers)
    :return: A list of voxel centers
    """
    # Convert the mesh to a voxel grid
    voxels = mesh.voxelized(pitch)
    
    # Get the voxel grid as coordinates of the voxel centers
    voxel_centers = voxels.points
    
    return voxel_centers


def create_voxel_cube(x, y, z, size):
    """
    Create the vertices and faces for a cube centered at (x, y, z) with the given size.
    """
    # Half size for convenience
    half_size = size / 2

    # Define the vertices of the cube
    vertices = np.array([
        [x - half_size, y - half_size, z - half_size],  # bottom-left-back
        [x + half_size, y - half_size, z - half_size],  # bottom-right-back
        [x + half_size, y + half_size, z - half_size],  # top-right-back
        [x - half_size, y + half_size, z - half_size],  # top-left-back
        [x - half_size, y - half_size, z + half_size],  # bottom-left-front
        [x + half_size, y - half_size, z + half_size],  # bottom-right-front
        [x + half_size, y + half_size, z + half_size],  # top-right-front
        [x - half_size, y + half_size, z + half_size],  # top-left-front
    ])

    # Define the 12 triangular faces of the cube
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # back face
        [4, 5, 6], [4, 6, 7],  # front face
        [0, 1, 5], [0, 5, 4],  # bottom face
        [2, 3, 7], [2, 7, 6],  # top face
        [0, 3, 7], [0, 7, 4],  # left face
        [1, 2, 6], [1, 6, 5],  # right face
    ])

    return vertices, faces

def create_voxel_plot(voxel_centers, voxel_size=0.01):
    """
    Create a Plotly 3D plot of voxels represented as cubes with different colors.
    :param voxel_centers: A list of voxel center coordinates (x, y, z)
    :param voxel_size: The size of each voxel cube
    :return: A Plotly Mesh3d figure
    """
    all_vertices = []
    all_i = []
    all_j = []
    all_k = []
    all_colors = []
    offset = 0

    for idx, center in enumerate(voxel_centers):
        x, y, z = center
        vertices, faces = create_voxel_cube(x, y, z, voxel_size)

        # Add vertices for this cube
        all_vertices.extend(vertices)

        # Add face indices for this cube (adjusted for current vertex count)
        all_i.extend(faces[:, 0] + offset)
        all_j.extend(faces[:, 1] + offset)
        all_k.extend(faces[:, 2] + offset)

        # color_value = np.random.rand(3) 
        # voxel_color = [color_value] * len(vertices)  
        # all_colors.extend(voxel_color)  
        
        gray_value = np.random.rand()  
        voxel_color = [[gray_value, gray_value, gray_value]] * len(vertices) 
        all_colors.extend(voxel_color) 

        # Update offset for the next set of vertices
        offset += len(vertices)

    # Convert to numpy arrays
    all_vertices = np.array(all_vertices)
    all_i = np.array(all_i)
    all_j = np.array(all_j)
    all_k = np.array(all_k)
    all_colors = np.array(all_colors)  # Colors for each voxel's vertices

    # Create the 3D mesh trace with vertex colors
    trace = go.Mesh3d(
        x=all_vertices[:, 0],
        y=all_vertices[:, 1],
        z=all_vertices[:, 2],
        i=all_i,
        j=all_j,
        k=all_k,
        vertexcolor=all_colors,  # Set vertex colors
        opacity=1,
        flatshading=True
    )

    # Define the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode='data'  # Keep the natural aspect ratio
        ),
        margin=dict(l=0, r=0, t=0, b=0)  # No margins
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def plot_voxel_3d_shape_interface(file_path, pitch=0.01, voxel_size=0.01):
    """
    Plot a 3D voxel representation of the mesh as actual cubes in Streamlit using Plotly.
    :param file_path: Path to the 3D object file (.obj format)
    :param pitch: The size of the voxels (distance between centers)
    :param voxel_size: The size of each voxel cube
    """
    # Create a filename for saving the voxelized data (based on the original file name)
    voxel_file_path = f"outputs/data/_voxelized_pitch_{pitch}.npy"
    
    # Check if the voxelized data already exists
    if os.path.exists(voxel_file_path):
        # Load the saved voxel centers
        voxel_centers = np.load(voxel_file_path)
    else:
        # Load the mesh using Trimesh
        mesh = load_object_file(file_path)
        
        # Convert the mesh to voxels
        voxels = mesh.voxelized(pitch)
        voxel_centers = voxels.points
        
        # Save the voxel centers for future use
        np.save(voxel_file_path, voxel_centers)
    
    # Create the voxel plot
    fig = create_voxel_plot(voxel_centers, voxel_size)

    # Display the plot in Streamlit
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        st.plotly_chart(fig)


def create_techniques_tools_table(stage='prep1'):
    
    st.markdown("### Tools and techniques used")
    if stage == 'prep1':
        techniques_tools = [
            'Mesh Cleaning', 
            'Subdivision', 
            'Decimation (30%)',
            'Multiprocessing',
        ]

        descriptions = [
            'Remove degenerate (zero area) faces, duplicate, unreferenced and infinite (NaN) \
            vertices, fill holes (missing faces) and fix normals.',
            'Increase the number of triangles by splitting each triangle into smaller ones.',
            'Reduces the triangle count by 30%, simplifying the mesh while preserving the overall shape.',
            'Allocate the meshes in 8-16 processes to speed up the resampling process.',
        ]
        
    elif stage == 'prep2':
        techniques_tools = [
            'Isotropic Remeshing', 
            'Mesh Cleaning',
            'Multiprocessing',
        ]

        descriptions = [
            'Adjust triangle shapes and sizes through edge flipping, collapsing, relaxing, and refining, resulting \
                in a more uniform and well-proportioned mesh with better-quality triangles.',
            'After the remeshing procedure all the meshes were cleaned exactly like in the previous step.',
            'Allocate the meshes in 8-16 processes to speed up the remeshing process.',
        ]  
    
    elif stage == 'prep3':
        techniques_tools = [
            'Center at Origin', 
            'Scale Unit Cube', 
            'PCA Align.',
            'Moment Flipping',
            'Multiprocessing',
        ]

        descriptions = [
            'Reposition the mesh so that its center aligns with the origin, making it location-independent by \
                translating all vertices based on the centroid.',
            'Resize mesh to fit within a unit cube, using its largest bounding box dimension as a scaling factor.',
            'Align the mesh along its principal axes by rotating it based on the directions of maximum variation.',
            'Flip the mesh along any axis where its "heaviest" portions lie in the negative half.',
            'Allocate the meshes in 8-16 processes to speed up the normalization process.'
        ]
    
    
    elif stage == 'prep4':
        techniques_tools = [
            'Mesh Cleaning', 
            'Subdivision', 
            'Decimation',
            'Multiprocessing',
        ]

        descriptions = [
            'The total surface area of the 3D object.',
            'The maximum distance between any two points in the 3D object.',
            'Measures how much the shape deviates from being a perfect sphere.',
            'Measures how much the shape deviates from being a perfect sphere.',
        ]

    # Create a pandas DataFrame to organize the descriptors and their descriptions
    data = {'Tools/Techniques': techniques_tools, 'Description': descriptions}
    df = pd.DataFrame(data)

    # Display the table in Streamlit
    create_table_html(df)
    
    

def create_descriptor_table(type="global"):
    """
    Create and display a table with global descriptors and their descriptions.
    """
    if type == "global":
        # Define the global descriptors and their descriptions
        descriptors = [
            'Volume', 
            'Surface Area', 
            'Diameter', 
            'Eccentricity', 
            'Compactness', 
            'Rectangularity', 
            'Convexity', 
            'Sphericity', 
            'Elongation'
        ]

        descriptions = [
            'The total volume occupied by the 3D object.',
            'The total surface area of the 3D object.',
            'The maximum distance between any two points in the 3D object.',
            'Measures how much the shape deviates from being a perfect sphere.',
            'A ratio of volume to surface area that indicates how compact the shape is.',
            'The ratio of the object’s bounding box volume to its actual volume.',
            'The ratio of the convex hull volume to the actual volume of the object.',
            'Measures how spherical the object is, based on its volume and surface area.',
            'Measures how much the object is stretched along its principal axes.'
        ]

        # Create a pandas DataFrame to organize the descriptors and their descriptions
        data = {'Global Descriptor': descriptors, 'Description': descriptions}
        
    elif type == "local":
        # Define the global descriptors and their descriptions
        descriptors = [
            'A3', 
            'D1', 
            'D2', 
            'D3', 
            'D4', 
        ]

        descriptions = [
            'Angles between randomly chosen triplets of vertices, capture the local curvature and surface \
                smoothness or sharpness of the mesh. Samples: 4000.',
            'Distances from the centroid to random vertices, indicate how spread out or \
                concentrated the mesh is around its center. Samples: 4000.',
            "Distances between pairs of random vertices, reflect the object's overall shape \
                and point distribution. Samples: 4000.",
            'Areas of triangles formed by random vertices, prove insights into \
                local surface geometry and curvature. Samples: 4000.',
            'Volumes of tetrahedrons formed by random vertices, capture the \
                3D structural complexity of the mesh. Samples: 4000.',
        ]

        # Create a pandas DataFrame to organize the descriptors and their descriptions
        data = {'Global Descriptor': descriptors, 'Description': descriptions}
        
        
        
    elif type == "search":
        # Define the global descriptors and their descriptions
        descriptors = [
            'Euclidean distance', 
            "Earth Mover's dist.", 
            'Distance Weighting', 
        ]

        descriptions = [
            'Measure the shortest distance between two points in a multi-dimensional space. Direct \
                differences between the feature values of objects',
            'EMD, or Wasserstein distance to compare histograms. EMD measures how one histogram (distribution) \
                can be transformed into another.',
            "Standardize the distance values before combining them. Ensures that distances from different \
                descriptors contribute fairly to the overall similarity score.",
        ]

        # Create a pandas DataFrame to organize the descriptors and their descriptions
        data = {'Technique': descriptors, 'Description': descriptions}
        
    df = pd.DataFrame(data)

    # Display the table in Streamlit
    create_table_html(df)
    
    
def create_search_engine_table(stage="search"):
    """
    Create and display a table with global descriptors and their descriptions.
    """  

    st.markdown("### Techniques used")
    if stage == "search":
        # Define the global descriptors and their descriptions
        descriptors = [
            'Euclidean distance', 
            "Earth Mover's dist.", 
            'Distance Weighting', 
        ]

        descriptions = [
            'Measure the shortest distance between two points in a multi-dimensional space. Direct \
                differences between the feature values of objects',
            'EMD, or Wasserstein distance to compare histograms. EMD measures how one histogram (distribution) \
                can be transformed into another.',
            "Standardize the distance values before combining them. Ensures that distances from different \
                descriptors contribute fairly to the overall similarity score.",
        ]

        # Create a pandas DataFrame to organize the descriptors and their descriptions
        data = {'Technique': descriptors, 'Description': descriptions}
        
    df = pd.DataFrame(data)

    # Display the table in Streamlit
    create_table_html(df)
    

def create_spider_plots(df, class1, class2, descriptors):
    """
    Create two spider plots for two specific classes, showing the mean values
    of the selected descriptors for each class.
    
    :param df: The DataFrame containing the data.
    :param class1: The first class for comparison.
    :param class2: The second class for comparison.
    :param descriptors: The list of global descriptors to plot.
    """
    # Group by 'obj_class' and compute the mean of the descriptors
    grouped = df.groupby('obj_class')[descriptors].mean()

    # Extract the mean values for the two specified classes
    class1_data = grouped.loc[class1].values
    class2_data = grouped.loc[class2].values

    # Create a DataFrame for each class with descriptor names and values
    class1_df = pd.DataFrame({
        'Descriptor': descriptors,
        'Mean Value': class1_data
    })
    
    class2_df = pd.DataFrame({
        'Descriptor': descriptors,
        'Mean Value': class2_data
    })

    # Create the spider plot for the first class
    fig1 = px.line_polar(class1_df, r='Mean Value', theta='Descriptor', line_close=True,
                         title=f'Spider Plot: {class1}')
    fig1.update_traces(fill='toself')
    fig1.update_layout(
        width=260,  # Adjust the width
        height=260,  # Adjust the height
        margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins
    )

    # Create the spider plot for the second class
    fig2 = px.line_polar(class2_df, r='Mean Value', theta='Descriptor', line_close=True,
                         title=f'Spider Plot: {class2}')
    fig2.update_traces(fill='toself')
    fig2.update_layout(
        width=260,  # Adjust the width
        height=260,  # Adjust the height
        margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins
    )


    # Display the two spider plots side by side using Streamlit columns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        

def plot_local_desc(image_A3, image_D1, image_D2, image_D3, image_D4):
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_A3, caption="A3")
        st.image(image_D2, caption="D2")
        st.image(image_D4, caption="D4")
    with col2:
        st.image(image_D1, caption="D1")
        st.image(image_D3, caption="D3")
    
    return 1


def show_time_elapsed(stage='resampling'):
    times_df = pd.read_csv('outputs/data/times.csv')
    st.markdown(f"### ~ Process time: {times_df[stage].values[0]:.2f} sec  |  {(times_df[stage].values[0]/60):.2f} min ~")
    

def show_images_side_by_side(path1, path2):
    col1, col2, col3 = st.columns([8, 1, 8])
    with col1:
        st.image(path1, caption="Before norm.")
    with col3:
        st.image(path2, caption="After norm.")


def create_table_html(df):
    # Convert the DataFrame to HTML with additional CSS for better alignment
        styled_table = df.to_html(index=False, classes="table-style")

        # Add CSS to center-align the headers and adjust table appearance
        st.markdown("""
        <style>
        .table-style {
            width: 100%;
            border-collapse: collapse;
            margin: 0;
            padding: 0;
            table-layout: auto;
            font-family: Arial, sans-serif;
            font-size:14px;
        }
        .table-style th {
            background-color: #f2f2f2;
            text-align: left;
            padding: 5px;
        }
        .table-style td {
            text-align: left;
            padding: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display the table
        st.write(styled_table, unsafe_allow_html=True)
        
        
def show_single_image(img_path, dataset_path):
    col1, col2 = st.columns([3, 6])
    with col1:
        df = pd.read_csv(dataset_path)
        create_table_html(df)
    with col2:
        st.image(img_path, caption="Class Distribution", use_column_width=True)
    

def eval_interactive_scatter_plot(data_path):
    # Load data
    data = pd.read_csv(data_path) 

    # Create the interactive scatter plot
    fig = px.scatter(
        data,
        x='mean_recall_regular',
        y='mean_recall_enhanced',
        text='obj_class',  # Display class names on hover
    )

    # Add a line y=x for reference
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="red")
    )

    fig.update_traces(marker=dict(size=8, color='black'), textposition='top center')
    fig.update_layout(xaxis_title="Regular Recall", yaxis_title="Enhanced Recall")
    st.plotly_chart(fig)
    
    
def top_changes_in_recall(data_path):
    # Load data and calculate change in recall
    data = pd.read_csv('outputs/eval/combined_recalls.csv')
    data['recall_change'] = data['mean_recall_enhanced'] - data['mean_recall_regular']

    # Find top 10 classes with greatest positive and negative change
    top_10_positive_change = data.nlargest(5, 'recall_change')
    top_10_negative_change = data.nsmallest(5, 'recall_change').sort_values(by=['recall_change'], ascending=False)
    change_data = pd.concat([top_10_positive_change, top_10_negative_change]).round(2)

    fig = px.bar(
        change_data,
        x='obj_class',
        y='recall_change',
        color='recall_change',
        color_continuous_scale='reds',
        text='recall_change'
    )

    fig.update_layout(xaxis_title="Class", yaxis_title="Change in Recall")
    st.plotly_chart(fig)
        

def create_donut_charts(value1, value2):
    """
    Create two side-by-side donut pie charts with Plotly showing input values as percentages at the center.
    
    Parameters:
        value1 (float): Value for the first chart (0-100).
        value2 (float): Value for the second chart (0-100).
    """
    # Ensure values are capped at 100 for valid percentage representation
    value1 = min(value1, 100)
    value2 = min(value2, 100)

    # Data for the charts
    chart1_data = [value1, 100 - value1]
    chart2_data = [value2, 100 - value2]

    # Labels for the charts
    labels = ['Value', '']  # The second label is empty to hide it

    # Create first donut chart
    fig1 = go.Figure(data=[go.Pie(
        labels=labels,
        values=chart1_data,
        hole=0.75,
        marker_colors=['#b00000', '#f7f7f7'],
        textinfo='none',  # Hide text on the slices
        sort=False,  # Keep slices in the same position
        rotation=0  # Start slices at the same position
    )])
    fig1.update_layout(
        title_text="Regular search",
        showlegend=False,
        annotations=[dict(
            text=f'{value1}%',  # Display percentage at the center
            x=0.5, y=0.5, font_size=25, showarrow=False
        )],
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Create second donut chart
    fig2 = go.Figure(data=[go.Pie(
        labels=labels,
        values=chart2_data,
        hole=0.75,
        marker_colors=['#b00000', '#f7f7f7'],
        textinfo='none',  # Hide text on the slices
        sort=False,  # Keep slices in the same position
        rotation=0  # Start slices at the same position
    )])
    fig2.update_layout(
        title_text="Enhanced search",
        showlegend=False,
        annotations=[dict(
            text=f'{value2}%',  # Display percentage at the center
            x=0.5, y=0.5, font_size=25, showarrow=False
        )],
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Display charts side by side in Streamlit
    col1, col2, col3 = st.columns([2,1.2,2])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col3:
        st.plotly_chart(fig2, use_container_width=True)


def display_custom_font_text(text, font_family="Brush Script MT"):
    """
    Display text with a custom font in Streamlit.

    Parameters:
        text (str): The text to display.
        font_url (str): URL to import the custom font (e.g., from Google Fonts or another source).
        font_family (str): The name of the font family to use.
    """
    # HTML and CSS code for custom font
    custom_html = f"""
    <style>
    .custom-font {{
        font-family: '{font_family}', sans-serif;
        font-size: 64px;
        color: black;
    }}
    </style>
    <div class="custom-font">{text}</div>
    """
    st.markdown(custom_html, unsafe_allow_html=True)

    
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


def presentation():
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #++++++++++++++++++++ PREPROCESSING +++++++++++++++++++++#
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    st.title("Classes and objects")
    show_single_image("outputs/plots/class_distribution_2.png", "outputs/data/class_distribution_2.csv")
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #++++++++++++++++++++ PREPROCESSING +++++++++++++++++++++#
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    st.title("Preprocessing")
    
    # -------------------- INITIAL DATA -------------------- #
    st.markdown("## 1. Original data")
    original_shapes_data_df = load_data('outputs/shapes_data.csv')
    original_global_statistics_df = load_data('outputs/data/global_statistics_original.csv')
    
    # create_techniques_tools_table(stage='prep1')
    
    st.markdown("###### ")
    plot_histograms(original_shapes_data_df)
    show_global_statistics(original_global_statistics_df)
    
    plot_3d_shape_interface('datasets/dataset_original/Car/m1548.obj')
    st.markdown("---")
    
    # -------------------- RESAMPLED DATA -------------------- #
    st.markdown("## 2. Resampled data")
    original_shapes_data_df = load_data('outputs/shapes_data_resampled_cleaned.csv')
    original_global_statistics_df = load_data('outputs/data/global_statistics_resampled_cleaned.csv')
        
    create_techniques_tools_table(stage='prep1')
    show_time_elapsed('resampling')
    
    st.markdown("##### ")
    plot_histograms(original_shapes_data_df)
    show_global_statistics(original_global_statistics_df)
    plot_3d_shape_interface('datasets/dataset_original_resampled_cleaned/Car/m1548.obj')
    st.markdown("---")
    
    # -------------------- REMESHED DATA -------------------- #
    st.markdown("## 3. Remeshed data")
    original_shapes_data_df = load_data('outputs/shapes_data_remeshed_cleaned.csv')
    original_global_statistics_df = load_data('outputs/data/global_statistics_remeshed_cleaned.csv')
        
    create_techniques_tools_table(stage='prep2')
    show_time_elapsed('remeshing')
    
    st.markdown("##### ")
    plot_histograms(original_shapes_data_df)
    show_global_statistics(original_global_statistics_df)
    plot_3d_shape_interface('datasets/dataset_original_remeshed_cleaned/Car/m1548.obj')
    st.markdown("---")
    
    # -------------------- NORMALIZED DATA -------------------- #
    st.markdown("## 4. Normalized data")
    original_shapes_data_df = load_data('outputs/shapes_data_normalized.csv')
    original_global_statistics_df = load_data('outputs/data/global_statistics_remeshed_cleaned.csv')
        
    create_techniques_tools_table(stage='prep3')
    show_time_elapsed('normalization')
    
    st.markdown("##### ")
    st.markdown("### Mesh centroids before and after normalization ")
    st.markdown("###### ")
    show_images_side_by_side("outputs/plots/before_norm_centroids_5_final.png", "outputs/plots/after_norm_centroids_5_final.png")
    st.markdown("###### ")
    st.markdown("### Mesh Bounding Box Diagonal length before and after normalization ")
    st.markdown("###### ")
    show_images_side_by_side("outputs/plots/bb_before_norm.png", "outputs/plots/bb_after_norm.png")
    
    # st.markdown("##### ")
    
    # plot_histograms(original_shapes_data_df)
    # show_global_statistics(original_global_statistics_df)
    
    plot_3d_shape_interface('datasets/dataset_original_normalized/Car/m1548.obj')
    st.markdown("---")
    st.markdown("---")
    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #++++++++++++++++++ FEATURE EXTRACTION ++++++++++++++++++#
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    
    # -------------------- GLOBAL DESCRIPTORS -------------------- #
    st.title("Feature Extraction")
    
    st.markdown("## 1. Global Descriptors")
    create_descriptor_table("global")
    try:
        show_time_elapsed('global_desc') 
        st.markdown("##### ")
    except:
        print("Time for global descriptors does not exist")
    
    global_descriptors_df = pd.read_csv('outputs/data/global_descriptors_standardized.csv')
    class1 = 'Car'
    class2 = 'House'
    descriptors = ['surface_area', 'diameter', 'eccentricity', 'compactness', 'rectangularity']
    st.markdown("### Example of global descriptor distribution (Car vs House)")
    create_spider_plots(global_descriptors_df, class1, class2, descriptors)
    
    st.markdown("### ")
    st.markdown("### Solution for the watertight meshes: Voxelization")
    plot_voxel_3d_shape_interface('datasets/dataset_original_normalized/Car/m1548.obj')
    st.markdown("---")
    
    # -------------------- LOCAL DESCRIPTORS -------------------- #
    st.markdown("## 2. Shape Property Descriptors")
    create_descriptor_table("local")
    
    try:
        show_time_elapsed('local_desc') 
        st.markdown("##### ")
    except:
        print("Time for local descriptors does not exist")
    
    st.markdown("### Example of shape property descriptor histogram")
    a_3 = "local_desc_plots/Bicycle/A3_plot.png"
    d_1 = "local_desc_plots/Bicycle/D1_plot.png"
    d_2 = "local_desc_plots/Bicycle/D2_plot.png"
    d_3 = "local_desc_plots/Bicycle/D3_plot.png"
    d_4 = "local_desc_plots/Bicycle/D4_plot.png"

    plot_local_desc(a_3, d_1, d_2, d_3, d_4)
    st.markdown("---")
    st.markdown("---")
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #+++++++++++++++++++++ SEARCH ENGINE ++++++++++++++++++++#
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    st.title("Search Engine")
    create_search_engine_table(stage='search')
    
    st.markdown("### Search engine video")
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.video("https://www.youtube.com/watch?v=XFmuaRAbRew")
    
    st.page_link("https://www.youtube.com/watch?v=XFmuaRAbRew", label="video link", icon="🔗")

    st.markdown("---")
    st.markdown("---")
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #++++++++++++++++++++++ EVALUATION ++++++++++++++++++++++#
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    st.title("Evaluation")

    st.markdown("### Regular vs. Enhanced Recall by Class")
    recall_data = 'outputs/eval/combined_recalls.csv'
    eval_interactive_scatter_plot(recall_data)
    st.markdown("##### ")
    
    st.markdown("### Top 5 Increases and Decreases in Recall")
    top_changes_in_recall(recall_data)
    st.markdown("##### ")
    
    st.markdown("### Total recall: regular vs enhanced (first 10 objects)")
    create_donut_charts(41.57, 63.52)

    st.markdown("---")
    display_custom_font_text("The end")




