
import numpy as np
import os
import pyvista as pv
from PIL import Image
from skimage import measure



def create_mesh_pyvista2(segmentation, labels=None, spacing=(1.0, 1.0, 1.0), flip_z=False):
    """
    Convert segmentation to a mesh using Marching Cubes, with label selection or merging.
    Each label gets a unique color.
    :param segmentation: Multi-label segmentation (3D numpy array)
    :param labels: Single label or list of labels to include (default: all labels)
    :param spacing: Voxel spacing for scaling
    :return: PyVista mesh object
    """
    # Define a color map (you can adjust these colors as needed)
    label_colors = {
        1: "red",
        2: "green",
        3: "blue",
        4: "yellow",
        5: "purple",
        6: "cyan",
        7: "magenta",
        # Add more colors for more labels
    }

    meshes = []

    if labels is not None:
        for label in labels:
            # Create a binary mask for the selected label
            # mask = segmentation == label
            if labels is not None:
                # Create a binary mask for the selected/merged labels
                mask = np.isin(segmentation, labels)
            else:
                # Use all non-zero labels
                mask = segmentation > 0

            if mask.sum() == 0:
                continue  # Skip labels that are not present in the data

            # Perform Marching Cubes on the binary mask
            verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0)
            verts *= np.array(spacing)  # Apply spacing

            if flip_z:
                verts[:, 2] = -verts[:, 2]

            # Convert to PyVista mesh
            faces = np.hstack([[3] + list(face) for face in faces])  # Format for PyVista (triangle connectivity)
            mesh = pv.PolyData(verts, faces)
            meshes.append((mesh, label_colors.get(label, "gray")))  # Default to gray if label not in dict

            # mesh.point_arrays["Label"] = np.full(mesh.n_points, label)  # Add label as point data
            # meshes.append((mesh, label_colors.get(label, "gray")))  # Default to gray if label not in dict

    # Combine all meshes into one (with different colors for each label)
    combined_mesh = pv.PolyData()
    for mesh, color in meshes:
        combined_mesh += mesh
        # combined_mesh.point_arrays["Color"] = np.array([label_colors.get(int(label), "gray") for label in mesh.point_arrays["Label"]])

    return combined_mesh


def create_point_cloud_pyvista(segmentation, labels=None, spacing=(1.0, 1.0, 1.0), flip_z=False):
    """
    Convert segmentation data to a point cloud with optional label filtering.
    Each label gets a unique color.
    :param segmentation: 3D numpy array of segmentation data
    :param labels: List of labels to visualize (default: all labels)
    :param spacing: Voxel spacing for scaling
    :param flip_z: If True, flip Z-axis values
    :return: List of (points, color) for each label
    """
    # Define a color map for labels
    label_colors = {
        1: "red",
        2: "green",
        3: "blue",
        4: "yellow",
        5: "purple",
        6: "cyan",
        7: "magenta",
        # Add more colors as needed
    }

    points_list = []

    # If labels are not specified, use all unique labels in the segmentation
    if labels is None:
        labels = np.unique(segmentation)
        labels = labels[labels > 0]  # Ignore background (label = 0)

    for label in labels:
        # Find the points corresponding to this label
        indices = np.argwhere(segmentation == label)
        if flip_z:
            indices[:, 2] = -indices[:, 2]  # Flip the Z-axis

        # Scale points based on spacing
        points = indices * np.array(spacing)

        # Store points and corresponding color
        color = label_colors.get(label, "gray")  # Default to gray for unspecified labels
        points_list.append((points, color))

    return points_list


def visualize_4d_and_save_pyvista(data_4d, labels=None, spacing=(1.0, 1.0, 1.0), output_dir="output", gif_name="animation.gif", 
                                  flip_z=False, method='mesh'):
    """
    Visualize 4D segmentation data as 3D+t animation using PyVista, and save as PNGs and a GIF.
    :param data_4d: 4D numpy array (X, Y, Z, T)
    :param labels: Single label or list of labels to include (default: all labels)
    :param spacing: Voxel spacing for scaling
    :param output_dir: Directory to save the output PNGs
    :param gif_name: Name of the output GIF
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    frames = []

    # Create a PyVista Plotter with offscreen rendering enabled
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter.background_color = "white"

    for t in range(data_4d.shape[-1]):  # Iterate through time steps
        print(f"Processing time step {t + 1}/{data_4d.shape[-1]}")

        plotter.clear()  # Clear the scene for the next frame

        segmentation = data_4d[..., t]  # Get 3D segmentation for this time step
        # mesh = create_mesh_pyvista(segmentation, labels, spacing)
        if method == "point_cloud":
            points_list = create_point_cloud_pyvista(segmentation, labels, spacing, flip_z)
            # Add each label's points to the plotter
            for points, color in points_list:
                plotter.add_points(points, color=color, point_size=4, 
                                   render_points_as_spheres=True, opacity=0.95)

        elif method == "mesh":  
            mesh = create_mesh_pyvista2(segmentation, labels, spacing, flip_z)
            plotter.add_mesh(mesh, 
                            color="white", 
                            opacity=0.10, 
                            show_edges=True,
                            edge_color="black",
                            line_width=1.0,
                            )

        # Adjust camera view (optional, tweak if needed)
        # plotter.view_isometric()

        # Save the frame as a PNG
        png_path = os.path.join(output_dir, f"frame_{t:03d}.png")
        plotter.screenshot(png_path)
        frames.append(png_path)

    plotter.close()

    # Create a GIF from saved PNGs
    gif_frames = [Image.open(frame) for frame in frames]
    gif_frames[0].save(
        os.path.join(output_dir, gif_name),
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,  # Frame duration in milliseconds
        loop=0
    )
    print(f"Animation saved as {os.path.join(output_dir, gif_name)}")