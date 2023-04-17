import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import slideio
import torch
from PIL import Image
from tqdm import tqdm
import re

from models.model import get_models
from utils.utils import (bgr_format, get_driver, get_scaling, 
                        save_hdf5, save_tile_preview, threshold,
                        save_qupath_annotation)


import concurrent.futures
import time

parser = argparse.ArgumentParser(description='Feature extraction')

parser.add_argument('--slide_path', help='path of slides to extract features from', default='/mnt/slides/augsburg_data', type=str)
parser.add_argument('--save_path', help='path to save everything', default='/home/ubuntu/run/0', type=str)
parser.add_argument('--file_extension', help='file extension the slides are saved under, e.g. tiff', default='.tiff', type=str)
parser.add_argument('--models', help='select model ctranspath, retccl, all', nargs='+', default=['retccl'], type=str)
parser.add_argument('--scene_list', help='list of scene(s) to be extracted', nargs='+', default=[0], type=int)
parser.add_argument('--save_patch_images', help='True if each patch should be saved as an image', default=True, type=bool)
parser.add_argument('--patch_size', help='Patch size for saving', default=256, type=int)
parser.add_argument('--white_thresh', help='if all RGB pixel values are larger than this value, the pixel is considered as white/background', default=170, type=int)
parser.add_argument('--black_thresh', help='if all RGB pixel values are smaller or equal than this value, the pixel is considered as black/background', default=0, type=str)
parser.add_argument('--invalid_ratio_thresh', help='maximum acceptable amount of background', default=0.5, type=float)
parser.add_argument('--edge_threshold', help='canny edge detection threshold. if smaller than this value, patch gets discarded', default=4, type=int)
parser.add_argument('--resolution_in_mpp', help='resolution in mpp, usually 10x= 1mpp, 20x=0.5mpp, 40x=0.25, ', default=0, type=float)
parser.add_argument('--downscaling_factor', help='only used if >0, overrides manual resolution. needed if resolution not given', default=1, type=float)
parser.add_argument('--save_tile_preview', help='set True if you want nice pictures', default=True, type=bool)
parser.add_argument('--preview_size', help='size of tile_preview', default=20000, type=int)
parser.add_argument('--exctraction_list', help='if only a subset of the slides should be extracted save their names in a csv', default="", type=str)
parser.add_argument('--save_qupath_annotation', help='set True if you want nice qupath annotations', default=True, type=bool)

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get slide files based on the provided path and file extension
    slide_files = sorted(Path(args.slide_path).glob(f'**/*{args.file_extension}'))
    
    if bool(args.exctraction_list) is not False:
        to_extract=pd.read_csv(args.exctraction_list).iloc[:, 0].tolist()
        slide_files = [file for file in slide_files if file.stem in to_extract]

    #filter out slide files using RegEx
    #slide_files = [file for file in slide_files if not re.search('_CR_|_CL_', str(file))]
    
    # Get model dictionaries
    model_dicts = get_models(args.models)

    # Get the driver for the slide file extension
    driver = get_driver(args.file_extension)

    # Create output directory
    output_path = Path(args.save_path) / 'h5_files'
    output_path.mkdir(parents=True, exist_ok=True)

    # Process models
    for model in model_dicts:
        model_name = model['name']
        save_dir = (Path(args.save_path) / 'h5_files' /
                    f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal')

        # Create save directory for the model
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of argument names and values
        arg_dict = vars(args)

        # Write the argument dictionary to a text file
        with open(save_dir / 'config.yml', 'w') as f:
            for arg_name, arg_value in arg_dict.items():
                f.write(f"{arg_name}: {arg_value}\n")

    # Create directories
    if args.save_tile_preview:
        tile_path=(Path(args.save_path) / f'tiling_previews_{args.patch_size}px_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal')
        tile_path.mkdir(parents=True, exist_ok=True)
    else:
        tile_path = None

    if args.save_qupath_annotation:
        annotation_path=(Path(args.save_path) / f'qupath_annotation_{args.patch_size}px_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal')
        annotation_path.mkdir(parents=True, exist_ok=True)
    else:
        annotation_path = None
    # Process slide files
    start = time.perf_counter()
    for slide_file in tqdm(slide_files,position=0,leave=False,desc='slides'):
        slide = slideio.Slide(str(slide_file), driver)
        slide_name = slide_file.stem
        extract_features(slide, slide_name, model_dicts,device,args,tile_path, annotation_path)

    end = time.perf_counter()
    elapsed_time = end - start

    print("Time taken: ", elapsed_time, "seconds")

def extract_features(slide, slide_name, model_dicts,device,args,tile_path, annotation_path): 
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
    coords = pd.DataFrame({'scn' : [], 'x' : [], 'y' : []}, dtype=int) 

    if args.save_patch_images:
        (Path(args.save_path) / 'patches'/ slide_name).mkdir(parents=True, exist_ok=True)

    #iterate over scenes of the slides
    for scn in args.scene_list: #range(slide.num_scenes):
            
        scene=slide.get_scene(scn)
        scaling=get_scaling(args,scene.resolution[0])

        #read the scene in the desired resolution
        wsi=scene.read_block(size=(int(scene.size[0]//scaling), int(scene.size[1]//scaling)))

        #revert the flipping
        wsi=np.transpose(wsi, (1, 0, 2))
        
        #check if RGB or BGR is used and adapt
        if bgr_format(slide.raw_metadata):
            wsi = wsi[..., ::-1]
            #print("Changed BGR to RGB!")
        
        # Define the main loop that processes all patches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            def process_row(wsi, scn, x, args):

                patches = []
                patches_coords = pd.DataFrame()

                for y in range(0, wsi.shape[1], args.patch_size):

                    #check if a full patch still 'fits' in y direction
                    if y+args.patch_size > wsi.shape[1]:
                        continue
                    
                    #extract patch
                    patch = wsi[x:x+args.patch_size, y:y+args.patch_size, :]

                    #threshold checks if it meets canny edge detection, white and black pixel criteria
                    if threshold(patch,args):
                        # Extract patch
                        patches.append(patch)
                        patches_coords = pd.concat([patches_coords, pd.DataFrame({'scn': [scn], 'x': [x], 'y': [y]})], ignore_index=True)
                
                return patches, patches_coords

            def patches_to_feature(patches, coords, model_dicts):
                feats = {model_dict["name"]: [] for model_dict in model_dicts}

                for patch, (_, (_, x, y)) in zip(patches, coords.iterrows()):
                    im = Image.fromarray(patch)

                    if args.save_patch_images:
                        im.save(Path(args.save_path) / 'patches' / slide_name/ f'{slide_name}_patch_{scn}_{x}_{y}.png')

                    #model inference on single patches
                    with torch.no_grad():
                        for model_dict in model_dicts:
                            model=model_dict['model']
                            transform=model_dict['transforms']
                            model_name=model_dict['name']
                            img_t = transform(im)
                            batch_t = img_t.unsqueeze(0).to(device)
                            features = model(batch_t)
                            feats[model_name].append(features)

                return feats
                    
            #iterate over x (width) of scene
            for x in tqdm(range(0, wsi.shape[0], args.patch_size),position=1,leave=False,desc=slide_name+"_"+str(scn)):

                #check if a full patch still 'fits' in x direction
                if x+args.patch_size > wsi.shape[0]:
                    continue
            
                future = executor.submit(process_row, wsi, scn, x, args)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches, patches_coords = future.result()
                if len(patches_coords) > 0:
                    coords = pd.concat([coords, patches_coords], ignore_index=True)
                    patch_feats = patches_to_feature(patches, patches_coords, model_dicts)
                    for key in patch_feats.keys():
                        feats[key].extend(patch_feats[key])
           
        #saves tiling preview on slide in desired size
        if args.save_tile_preview:
            save_tile_preview(args, slide_name, scn, wsi,coords, tile_path)

        if args.save_qupath_annotation:
            save_qupath_annotation(args, slide_name, scn, coords, annotation_path)

    # Write data to HDF5
    save_hdf5(args, slide_name, coords, feats)
           

    
if __name__=='__main__':
    args = parser.parse_args() 
    main(args)
    
                                         
