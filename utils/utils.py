import json
import xml.etree.ElementTree as ET

import cv2
import h5py
import numpy as np
import torch
from pathlib import Path

def bgr_format(xml_string):  # check if BGR or RGB
    root = ET.fromstring(xml_string)
    pixel_type_elem = root.findall(".//PixelType")
    return 'bgr' in pixel_type_elem[0].text.lower() if pixel_type_elem is not None else False


def get_driver(extension_name):
    if extension_name in ['.tiff', '.jpg', '.jpeg', '.png', '.tif']:
        return 'GDAL'
    elif extension_name == '':
        return 'DICOM'
    else:
        return extension_name.replace('.', '').upper()


def get_scaling(args, mpp_resolution_slide):
    if args.downscaling_factor > 0:
        return args.downscaling_factor
    else:
        return args.resolution_in_mpp/(mpp_resolution_slide*1e06)


def threshold(patch, args):
    whiteish_pixels = np.count_nonzero((patch[:, :, 0] > args.white_thresh) & (
        patch[:, :, 1] > args.white_thresh) & (patch[:, :, 2] > args.white_thresh))
    black_pixels = np.count_nonzero((patch[:, :, 0] <= args.black_thresh) & (
        patch[:, :, 1] <= args.black_thresh) & (patch[:, :, 2] <= args.black_thresh))

    # Compute the ratio of foreground pixels to total pixels in the patch
    invalid_ratio = (whiteish_pixels + black_pixels) / \
        (patch.shape[0] * patch.shape[1])

    if invalid_ratio <= args.invalid_ratio_thresh:

        edge = cv2.Canny(patch, 40, 100)
        if np.max(edge) > 0:
            edge = np.mean(edge) * 100 / np.max(edge)
        else:
            edge = 0

        if (edge < args.edge_threshold) or np.isnan(edge):
            return False
        else:
            return True

    else:
        return False



def save_tile_preview(args, slide_name, scn, preview_im):
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
    preview_im.save(Path(args.save_path) / 'tiling_previews' / f'{slide_name}_{scn}.png')


def save_hdf5(args, slide_name, model_name, coords, feats):
    """
    Save the extracted features and coordinates to an HDF5 file.

    Args:
        args (argparse.Namespace): Arguments containing various processing parameters.
        slide_name (str): Name of the slide file.
        model_name (str): Name of the model used for feature extraction.
        coords (pd.DataFrame): Coordinates of the extracted patches.
        feats (list): Extracted features.

    Returns:
        None
    """
    with h5py.File(Path(args.save_path) / 'h5_files' / f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal' / f'{slide_name}.h5', 'w') as f:
        f['coords'] = coords.astype('float64')
        f['feats'] = torch.cat(feats, dim=0).cpu().numpy()
        f['args'] = json.dumps(vars(args))
        f['model_name'] = model_name