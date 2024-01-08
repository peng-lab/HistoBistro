import argparse
import concurrent.futures
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import slideio
import torch
from PIL import Image
# import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


from dataset import SlideDataset
from models.model import get_models
from utils.utils import (bgr_format, get_driver, get_scaling, save_hdf5,
                         save_qupath_annotation, save_tile_preview, threshold)

parser = argparse.ArgumentParser(description="Feature extraction")

# 20-041_K6_vK
# 20-069_K3_vK
# 20-009_K6_vK

parser.add_argument(
    "--slide_path",
    help="path of slides to extract features from",
    default="/mnt/ceph_vol/raw_data/2021",
    type=str,
)
parser.add_argument(
    "--save_path",
    help="path to save everything",
    default="/mnt/ceph_vol/features/test/",
    type=str,
)
parser.add_argument(
    "--file_extension",
    help="file extension the slides are saved under, e.g. tiff",
    default=".czi",
    type=str,
)
parser.add_argument(
    "--models",
    help="select model ctranspath, retccl, all",
    nargs="+",
    default=[],
    type=str,
)
parser.add_argument(
    "--scene_list",
    help="list of scene(s) to be extracted",
    nargs="+",
    default=[0],
    type=int,
)
parser.add_argument("--patch_size", help="Patch size for saving", default=512, type=int)
parser.add_argument(
    "--white_thresh",
    help="if all RGB pixel values are larger than this value, the pixel is considered as white/background",
    default=[175, 190, 178],
    nargs='+',
    type=int,
)
parser.add_argument(
    "--black_thresh",
    help="if all RGB pixel values are smaller or equal than this value, the pixel is considered as black/background",
    default=0,
    type=str,
)
parser.add_argument(
    "--invalid_ratio_thresh",
    help="maximum acceptable amount of background",
    default=0.3,
    type=float,
)
parser.add_argument(
    "--edge_threshold",
    help="canny edge detection threshold. if smaller than this value, patch gets discarded",
    default=1,
    type=int,
)
parser.add_argument(
    "--resolution_in_mpp",
    help="resolution in mpp, usually 10x= 1mpp, 20x=0.5mpp, 40x=0.25, ",
    default=0,
    type=float,
)
parser.add_argument(
    "--downscaling_factor",
    help="only used if >0, overrides manual resolution. needed if resolution not given",
    default=0,
    type=float,
)
parser.add_argument(
    "--save_tile_preview",
    help="set True if you want nice pictures",
    action='store_true',
)

parser.add_argument(
    "--save_patch_images",
    help="True if each patch should be saved as an image",
    action='store_true',
)
parser.add_argument(
    "--preview_size", help="size of tile_preview", default=4096, type=int
)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument(
    "--exctraction_list",
    help="if only a subset of the slides should be extracted save their names in a csv",
    default=None, #"/home/ubuntu/idkidc/extraction_list.csv"
    type=str,
)  #
parser.add_argument(
    "--save_qupath_annotation",
    help="set True if you want nice qupath annotations",
    default=False,
    type=bool,
)
parser.add_argument(
    "--calc_thresh",
    help="darker colours than this are considered calc",
    default=[40, 40, 40],
    nargs='+',
    type=int,
)
parser.add_argument(
    "--split",
    help="(k,n): split slides into n distinct chunks and process number k. (0,1) for all slides at once. E.g. one urn with (0,2) and one with (1,2) to split data.",
    default=[0, 1],
    nargs='+',
    type=int,
)


def main(args):
    """
    Args:
    args: argparse.Namespace, containing the following attributes:
    - slide_path (str): Path to the slide files.
    - save_path (str): Path where to save the extracted features.
    - file_extension (str): File extension of the slide files (e.g., '.czi').
    - models (list): List of models to use for feature extraction.
    - scene_list (list): List of scenes to process.
    - save_patch_images (bool): Whether to save each patch as an image.
    - patch_size (int): Size of the image patches to process.
    - white_thresh (int): Threshold for considering a pixel as white/background (based on RGB values).
    - black_thresh (int): Threshold for considering a pixel as black/background (based on RGB values).
    - invalid_ratio_thresh (float): Threshold for invalid ratio in patch images.
    - edge_threshold (int): Canny edge detection threshold. Patches with values smaller than this are discarded.
    - resolution_in_mpp (float): Resolution in microns per pixel (e.g., 10x=1mpp, 20x=0.5mpp, 40x=0.25).
    - downscaling_factor (float): Downscaling factor for the images; used if >0, overrides manual resolution.
    - save_tile_preview (bool): Set to True if you want to save tile preview images.
    - preview_size (int): Size of tile_preview images.
    - extraction_list (str): Path to csv file containing the list of slides to be extracted (optional).
    - save_qupath_annotation (bool): Set to True if you want to save nice QuPath annotations (optional).

    Returns:
    None
    """

    # Set device to GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get slide files based on the provided path and file extension
    slide_files = sorted(Path(args.slide_path).glob(f"**/*{args.file_extension}"))

    if bool(args.exctraction_list) is not False:
        to_extract = pd.read_csv(args.exctraction_list).iloc[:, 0].tolist()
        slide_files = [file for file in slide_files if file.name in to_extract]

    # filter out slide files using RegEx
    #slide_files = sorted([
    #    file for file in slide_files if re.search("-DX", str(file))
    #])
    chunk_len=math.ceil(len(slide_files)/args.split[1])
    start=args.split[0]*chunk_len
    end=min(start+chunk_len,len(slide_files))
    slide_files=slide_files[start:end]

    # Get model dictionaries
    model_dicts = get_models(args.models)

    # Get the driver for the slide file extension
    driver = get_driver(args.file_extension)

    # Create output directory
    output_path = Path(args.save_path) / "h5_files"
    output_path.mkdir(parents=True, exist_ok=True)

    # Process models
    for model in model_dicts:
        model_name = model["name"]
        save_dir = (
            Path(args.save_path)
            / "h5_files"
            / f"{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal"
        )

        # Create save directory for the model
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of argument names and values
        arg_dict = vars(args)

        # Write the argument dictionary to a text file
        with open(save_dir / "config.yml", "w") as f:
            for arg_name, arg_value in arg_dict.items():
                f.write(f"{arg_name}: {arg_value}\n")

    # Create directories
    if args.save_tile_preview:
        tile_path = (
            Path(args.save_path)
            / f"tiling_previews_{args.patch_size}px_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal"
        )
        tile_path.mkdir(parents=True, exist_ok=True)
    else:
        tile_path = None

    if args.save_qupath_annotation:
        annotation_path = (
            Path(args.save_path)
            / f"qupath_annotation_{args.patch_size}px_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal"
        )
        annotation_path.mkdir(parents=True, exist_ok=True)
    else:
        annotation_path = None
    # Process slide files
    start = time.perf_counter()
    for slide_file in tqdm(slide_files, position=0, leave=False, desc="slides"):
        slide = slideio.Slide(str(slide_file), driver)
        slide_name = slide_file.stem
        extract_features(
            slide, slide_name, model_dicts, device, args, tile_path, annotation_path
        )

    end = time.perf_counter()
    elapsed_time = end - start

    print("Time taken: ", elapsed_time, "seconds")


def process_row(
    wsi: np.array, scn: int, x: int, args: argparse.Namespace, slide_name: str
):
    """
    Process a row of a whole slide image (WSI) and extract patches that meet the threshold criteria.

    Parameters:
    wsi (numpy.ndarray): The whole slide image as a 3D numpy array (height, width, color channels).
    scn (int): Scene number of the WSI.
    x (int): X coordinate of the patch in the WSI.
    args (argparse.Namespace): Parsed command-line arguments.
    slide_name (str): Name of the slide.

    Returns:
    pd.DataFrame: A DataFrame with the coordinates of the patches that meet the threshold.
    """

    patches_coords = pd.DataFrame()

    for y in range(0, wsi.shape[1], args.patch_size):
        # check if a full patch still 'fits' in y direction
        if y + args.patch_size > wsi.shape[1]:
            continue

        # extract patch
        patch = wsi[x : x + args.patch_size, y : y + args.patch_size, :]

        # threshold checks if it meets canny edge detection, white and black pixel criteria
        if threshold(patch, args):
            if args.save_patch_images:
                im = Image.fromarray(patch)
                im.save(
                    Path(args.save_path)
                    / "patches"
                    /str(args.downscaling_factor)
                    / slide_name
                    / f"{slide_name}_patch_{scn}_{x}_{y}.png"
                )

            patches_coords = pd.concat(
                [patches_coords, pd.DataFrame({"scn": [scn], "x": [x], "y": [y]})],
                ignore_index=True,
            )

    return patches_coords


def patches_to_feature(
    wsi: np.array, coords: pd.DataFrame, model_dicts: list[dict], device: torch.device
):
    feats = {model_dict["name"]: [] for model_dict in model_dicts}

    with torch.no_grad():
        for model_dict in model_dicts:
            model = model_dict["model"]
            transform = model_dict["transforms"]
            model_name = model_dict["name"]

            dataset = SlideDataset(wsi, coords, args.patch_size, transform)
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, num_workers=1, shuffle=False
            )

            for batch in dataloader:
                features = model(batch.to(device))
                feats[model_name] = feats[model_name] + (
                    features.cpu().numpy().tolist()
                )

    return feats


def extract_features(
    slide: slideio.py_slideio.Slide,
    slide_name: str,
    model_dicts: list[dict],
    device: torch.device,
    args: argparse.Namespace,
    tile_path: str,
    annotation_path: str,
):
    """
    Extract features from a slide using a given model.

    Args:
        slide (slideio.Slide): The slide object to process.
        slide_name (str): Name of the slide file.
        args (argparse.Namespace): Arguments containing various processing parameters.
        model_dict (dict): Dictionary containing the model, transforms, and model name.
        scene_list (list): List of scenes to process.
        device (torch.device): Device to perform computations on (CPU or GPU).
        tile_path (pathlib.Path): A Path object representing the path where the tile preview image will be saved.
        annotation_path (pathlib.Path): The path to the output directory.

    Returns:
        None
    """

    feats = {model_dict["name"]: [] for model_dict in model_dicts}
    coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)

    if args.save_patch_images:
        (Path(args.save_path) / "patches" / str(args.downscaling_factor)/slide_name).mkdir(
            parents=True, exist_ok=True
        )

    orig_sizes = []
    # iterate over scenes of the slides
    for scn in range(slide.num_scenes):
        scene_coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)
        scene = slide.get_scene(scn)
        orig_sizes.append(scene.size)
        try:
            scaling = get_scaling(args, scene.resolution[0])
        except Exception as e:
            print(e)
            print(f"Error determining resolution at slide ", slide_name, scn)
            break
        # read the scene in the desired resolution
        wsi = scene.read_block(
            size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling))
        )

        # revert the flipping
        # wsi=np.transpose(wsi, (1, 0, 2))

        # check if RGB or BGR is used and adapt
        #if bgr_format(slide.raw_metadata):
        #    wsi = wsi[..., ::-1]
            # print("Changed BGR to RGB!")

        # Define the main loop that processes all patches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            # iterate over x (width) of scene
            for x in tqdm(
                range(0, wsi.shape[0], args.patch_size),
                position=1,
                leave=False,
                desc=slide_name + "_" + str(scn),
            ):
                # check if a full patch still 'fits' in x direction
                if x + args.patch_size > wsi.shape[0]:
                    continue
                future = executor.submit(process_row, wsi, scn, x, args, slide_name)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches_coords = future.result()
                if len(patches_coords) > 0:
                    scene_coords = pd.concat(
                        [scene_coords, patches_coords], ignore_index=True
                    )
        coords = pd.concat([coords, scene_coords], ignore_index=True)

        if len(model_dicts)>0:
            patch_feats = patches_to_feature(wsi, scene_coords, model_dicts, device)
            for key in patch_feats.keys():
                feats[key].extend(patch_feats[key])

        # saves tiling preview on slide in desired size
        if args.save_tile_preview:
            save_tile_preview(args, slide_name, scn, wsi, coords, tile_path)

        if args.save_qupath_annotation:
            save_qupath_annotation(args, slide_name, scn, coords, annotation_path)

    # Write data to HDF5
    if len(model_dicts)>0:
        save_hdf5(args, slide_name, coords, feats, orig_sizes)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
