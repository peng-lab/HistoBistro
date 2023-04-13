import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


def bgr_format(xml_string):
    """
    Determine whether the image is in BGR or RGB format based on the PixelType element in the image metadata.

    Args:
    - xml_string: a string representing the image metadata in XML format.

    Returns:
    - A boolean value indicating whether the image is in BGR format (True) or not (False).
    """

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

    if extension_name in ['.tiff', '.jpg', '.jpeg', '.png', '.tif']:
        return 'GDAL'
    elif extension_name == '':
        return 'DICOM'
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
    whiteish_pixels = np.count_nonzero((patch[:, :, 0] > args.white_thresh) & (
        patch[:, :, 1] > args.white_thresh) & (patch[:, :, 2] > args.white_thresh))

    # Count the number of black pixels in the patch
    black_pixels = np.count_nonzero((patch[:, :, 0] <= args.black_thresh) & (
        patch[:, :, 1] <= args.black_thresh) & (patch[:, :, 2] <= args.black_thresh))

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


def save_tile_preview(args, slide_name, scn, preview_im, tile_path):
    """
    Save the tile preview image with the specified size.

    Args:
        args (argparse.Namespace): Arguments containing various processing parameters.
        slide_name (str): Name of the slide file.
        scn (int): Scene number.
        preview_im (PIL.Image.Image): The preview image to be saved.

    Returns:
        None
    """
    preview_size = int(args.preview_size)
    width, height = preview_im.size
    aspect_ratio = height / width

    if aspect_ratio > 1:
        new_height = preview_size
        new_width = int(preview_size / aspect_ratio)
    else:
        new_width = preview_size
        new_height = int(preview_size * aspect_ratio)

    preview_im = preview_im.resize((new_width, new_height))
    preview_im.save(tile_path /
                    f'{slide_name}_{scn}.png')


def save_hdf5(args, slide_name, coords, feats):
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
        with h5py.File(Path(args.save_path) / 'h5_files' / f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal' / f'{slide_name}.h5', 'w') as f:
            f['coords'] = coords.astype('float64')
            f['feats'] = torch.cat(features, dim=0).cpu().numpy()
            f['args'] = json.dumps(vars(args))
            f['model_name'] = model_name
