import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from PIL import Image

def bgr_format(xml_string):
    """
    Determine whether the image is in BGR or RGB format based on the PixelType element in the image metadata.

    Args:
    - xml_string: a string representing the image metadata in XML format.

    Returns:
    - A boolean value indicating whether the image is in BGR format (True) or not (False).
    """
    if xml_string == '':
        return False
    
    root = ET.fromstring(xml_string)
    pixel_type_elem = root.findall(".//PixelType")
    return 'bgr' in pixel_type_elem[0].text.lower() if pixel_type_elem is not None else False


def get_driver(extension_name):
    """
    Determine the driver to use for opening an image file based on its extension.

    Args:
    - extension_name: a string representing the file extension of the image file.

    Returns:
    - A string representing the driver to use for opening the image file.
    """

    if extension_name in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
        return 'GDAL'
    elif extension_name == '':
        return 'DCM'
    else:
        return extension_name.replace('.', '').upper()


def get_scaling(args, mpp_resolution_slide):
    """
    Determine the scaling factor to apply to an image based on the desired resolution in micrometers per pixel and the
    resolution in micrometers per pixel of the slide.

    Args:
    - args: a namespace containing the following attributes:
        - downscaling_factor: a float representing the downscaling factor to apply to the image.
        - resolution_in_mpp: a float representing the desired resolution in micrometers per pixel.
    - mpp_resolution_slide: a float representing the resolution in micrometers per pixel of the slide.

    Returns:
    - A float representing the scaling factor to apply to the image.
    """

    if args.downscaling_factor > 0:
        return args.downscaling_factor
    else:
        return args.resolution_in_mpp/(mpp_resolution_slide*1e06)


def threshold(patch, args):
    """
    Determine if a patch of an image should be considered invalid based on the following criteria:
    - The number of pixels with color values above a white threshold and below a black threshold should not exceed
    a certain ratio of the total pixels in the patch.
    - The patch should have significant edges.
    If these conditions are not met, the patch is considered invalid and False is returned.

    Args:
    - patch: a numpy array representing the patch of an image.
    - args: a namespace containing at least the following attributes:
        - white_thresh: a float representing the white threshold value.
        - black_thresh: a float representing the black threshold value.
        - invalid_ratio_thresh: a float representing the maximum ratio of foreground pixels to total pixels in the patch.
        - edge_threshold: a float representing the minimum edge value for a patch to be considered valid.

    Returns:
    - A boolean value indicating whether the patch is valid or not.
    """

    # Count the number of whiteish pixels in the patch
    whiteish_pixels = np.count_nonzero((patch[:, :, 0] > args.white_thresh[0]) & (
        patch[:, :, 1] > args.white_thresh[1]) & (patch[:, :, 2] > args.white_thresh[2]))

    # Count the number of black pixels in the patch
    black_pixels = np.count_nonzero((patch[:, :, 0] <= args.black_thresh) & (
        patch[:, :, 1] <= args.black_thresh) & (patch[:, :, 2] <= args.black_thresh))
    dark_pixels = np.count_nonzero((patch[:, :, 0] <= args.calc_thresh[0]) & (patch[:, :, 1] <= args.calc_thresh[1]) & (patch[:, :, 2] <= args.calc_thresh[2]))
    calc_pixels=dark_pixels-black_pixels

    if calc_pixels/(patch.shape[0] * patch.shape[1])>=0.05: #we always want to keep calc in!
        return True
    
    # Compute the ratio of foreground pixels to total pixels in the patch
    invalid_ratio = (whiteish_pixels + black_pixels) / \
        (patch.shape[0] * patch.shape[1])

    # Check if the ratio exceeds the threshold for invalid patches
    if invalid_ratio <= args.invalid_ratio_thresh:

        # Compute the edge map of the patch using Canny edge detection
        edge = cv2.Canny(patch, 40, 100)

        # If the maximum edge value is greater than 0, compute the mean edge value as a percentage of the maximum value
        if np.max(edge) > 0:
            edge = np.mean(edge) * 100 / np.max(edge)
        else:
            edge = 0

        # Check if the edge value is below the threshold for invalid patches or is NaN
        if (edge < args.edge_threshold) or np.isnan(edge):
            return False
        else:
            return True

    else:
        return False


def save_tile_preview(args, slide_name, scn, wsi, coords, tile_path):
    """
    Save the tile preview image with the specified size.

    Args:
        args (argparse.Namespace): A Namespace object that contains the arguments passed to the script.
        slide_name (str): A string representing the name of the slide file.
        scn (int): An integer representing the scene number.
        wsi (numpy.ndarray): A NumPy array representing the whole slide image.
        coords (pandas.DataFrame): A Pandas DataFrame containing the coordinates of the tiles.
        tile_path (pathlib.Path): A Path object representing the path where the tile preview image will be saved.

    Returns:
        None
    """

    # Draw bounding boxes for each tile on the whole slide image
    def draw_rect(wsi, x, y, size, color=[0, 0, 0], thickness=4):
        x2, y2 = x + size, y + size
        wsi[y:y+thickness, x:x+size, :] = color
        wsi[y:y+size, x:x+thickness, :] = color
        wsi[y:y+size, x2-thickness:x2, :] = color
        wsi[y2-thickness:y2, x:x+size, :] = color
  
    for _, [scene, x, y] in coords.iterrows():
        if scn==scene:
            draw_rect(wsi, y, x, args.patch_size)
        #cv2.rectangle(wsi.copy(), (x1, y1), (x2, y2), (0,0,0), thickness=4)

    # Convert NumPy array to PIL Image object
    preview_im = Image.fromarray(wsi)

    # Determine new dimensions of the preview image while maintaining aspect ratio
    preview_size = int(args.preview_size)
    width, height = preview_im.size
    aspect_ratio = height / width

    if aspect_ratio > 1:
        new_height = preview_size
        new_width = int(preview_size / aspect_ratio)
    else:
        new_width = preview_size
        new_height = int(preview_size * aspect_ratio)

    # Resize the preview image
    preview_im = preview_im.resize((new_width, new_height))

    # Save the preview image to disk
    preview_im.save(tile_path / f'{slide_name}_{scn}.png')

def save_qupath_annotation(args, slide_name, scn, coords, annotation_path):
    """
    Saves the QuPath annotation to a geojson file.

    Args:
        args (Namespace): Arguments for the script.
        slide_name (str): The name of the slide.
        scn (int): The SCN number of the slide.
        coords (pandas.DataFrame): The coordinates for the patches.
        annotation_path (pathlib.Path): The path to the output directory.

    Returns:
        None
    """
    
    # Function to create a single annotation feature
    def create_feature(coordinates, color):
        
        # Define the coordinates of the feature polygon
        x , y = coordinates[0], coordinates[1]
        top_left = coordinates
        top_right = [coordinates[0] + args.patch_size, coordinates[1]]
        bottom_right = [coordinates[0] + args.patch_size, coordinates[1] + args.patch_size]
        bottom_left = [coordinates[0], coordinates[1] + args.patch_size]
        coordinates = [top_left, top_right, bottom_right, bottom_left, top_left]

        # Create the feature dictionary with the specified properties
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": f'{x}, {y}', # random name
                    "color": color
                }
            }
        }
        return feature

    # Function to create a feature collection from a list of features
    def create_feature_collection(features):
        feature_collection = {
            "type": "FeatureCollection",
            "features": features
        }
        return feature_collection
    
    # Define the color of the annotation features
    color = [255, 0, 0]
    
    # Create a list of annotation features from the provided coordinates
    features = [create_feature([x, y], color) for _, [_, x, y] in coords.iterrows()]
    
    # Convert the list of features into a feature collection
    features = create_feature_collection(features)
    
    # Write the feature collection to a GeoJSON file
    with open(annotation_path / f'{slide_name}_{scn}.geojson', 'w') as annotation_file:
        # Write the dictionary to the file in JSON format
        json.dump(features, annotation_file)


def save_hdf5(args, slide_name, coords, feats,slide_sizes):
    """
    Save the extracted features and coordinates to an HDF5 file.
    Args:
        args (argparse.Namespace): Arguments containing various processing parameters.
        slide_name (str): Name of the slide file.
        coords (pd.DataFrame): Coordinates of the extracted patches.
        feats (dict): dictionary: modelname: extracted features
    Returns:
        None
    """
    for model_name, features in feats.items():
        if len(features)>0:
            with h5py.File(Path(args.save_path) / 'h5_files' / f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal' / f'{slide_name}.h5', 'w') as f:
                f['coords'] = coords.astype('float64')
                f['feats'] = features
                f['args'] = json.dumps(vars(args))
                f['model_name'] = model_name
                f['slide_sizes']=slide_sizes

            if len(np.unique(coords.scn))!=len(slide_sizes):
                print("SEMIWARNING, at least for one scene of ", slide_name, "no features were extracted, reason could be poor slide quality.")
        else:
            print("WARNING, no features extracted at slide", slide_name, "reason could be poor slide quality.")
