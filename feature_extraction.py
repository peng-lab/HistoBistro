import argparse
import json
from pathlib import Path

import cv2
import time
import h5py
import numpy as np
import pandas as pd
import slideio
import torch
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from models.model import get_models
from utils.utils import bgr_format, get_driver, get_scaling, threshold, append_to_csv_file

parser = argparse.ArgumentParser(description='Feature extraction')

parser.add_argument('--slide_path', help='path of slides to extract features from', default='/mnt/volume/raw_data/AIT', type=str)
parser.add_argument('--save_path', help='path to save everything', default='.', type=str)
parser.add_argument('--file_extension', help='file extension the slides are saved under, e.g. tiff', default='.czi', type=str)
parser.add_argument('--models', help='select model ctranspath, retccl, all', nargs='+', default=['kimianet'], type=str)
parser.add_argument('--scene_list', help='list of scene(s) to be extracted', nargs='+', default=[0,1], type=int)
parser.add_argument('--save_patch_images', help='True if each patch should be saved as an image', default=False, type=bool)
parser.add_argument('--patch_size', help='Patch size for saving', default=256, type=int)
parser.add_argument('--border_map', help='Set true or false for creating thumbnails with highlighted extracted patched', default=True, type=bool)
parser.add_argument('--white_thresh', help='if all RGB pixel values are larger than this value, the pixel is considered as white/background', default=170, type=int)
parser.add_argument('--black_thresh', help='if all RGB pixel values are smaller or equal than this value, the pixel is considered as black/background', default=0, type=str)
parser.add_argument('--invalid_ratio_thresh', help='True if each patch should be saved as an image', default=0.5, type=float)
parser.add_argument('--edge_threshold', help='canny edge detection threshold. if smaller than this value, patch gets discarded', default=4, type=int)
parser.add_argument('--resolution_in_mpp', help='resolution in mpp, usually 10x= 1mpp, 20x=0.5mpp, 40x=0.25, ', default=0, type=float)
parser.add_argument('--downscaling_factor', help='only used if >0, overrides manual resolution. needed if resolution not given', default=8, type=float)
parser.add_argument('--BGR_to_RGB', help='set True if your input has BGR format', default=True, type=bool)
parser.add_argument('--save_tile_preview', help='set True if you want nice pictures', default=True, type=bool)
parser.add_argument('--preview_size', help='size of tile_preview', default=4096, type=int)


def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    slide_files = Path(args.slide_path).glob(f'**/*{args.file_extension}')
    model_dicts= get_models(args.models)
    driver=get_driver(args.file_extension)

    output_path = Path(args.save_path)/'h5_files'
    output_path.mkdir(parents=True, exist_ok=True)

    for model in model_dicts:
        model_name=model['name']
        save_dir = (Path(args.save_path) / 'h5_files' / 
            f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal')
        
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of argument names and values
        arg_dict = vars(args)

        # Write the argument dictionary to a text file
        with open(save_dir /'config.yml', 'w') as f:
            for arg_name, arg_value in arg_dict.items():
                f.write(f"{arg_name}: {arg_value}\n")

    if args.save_tile_preview:
        (Path(args.save_path) / 'tiling_previews').mkdir(parents=True ,exist_ok=True)

    for slide_file in slide_files:
        slide = slideio.Slide(str(slide_file), driver)
        slide_name = slide_file.stem
        
        for model_dict in model_dicts:
            extract_features(slide, slide_name, args, model_dict,args.scene_list, device)


def extract_features(slide, slide_name,args, model_dict,scene_list,device): 
    time0=time.time()
    model=model_dict['model']
    transform=model_dict['transforms']
    model_name=model_dict['name']

    feats=[]
    coords = pd.DataFrame({'scn' : [], 'x' : [], 'y' : []}) 

    for scn in range(slide.num_scenes):
        wsi_copy=None
        scene=slide.get_scene(scn)
        scaling=get_scaling(args,scene.resolution[0])
        wsi=scene.read_block(size=(int(scene.size[0]//scaling), int(scene.size[1]//scaling)))
        wsi=np.transpose(wsi, (1, 0, 2))
        
        if bgr_format(slide.raw_metadata):
            wsi = wsi[..., ::-1]
            print("! Changed BGR to RGB !")

        #if args.save_tile_preview:
        #    wsi_copy=wsi.copy()

        feats, coords=process_wsi(wsi,args,transform,device,model,scn)
        # for x in tqdm(range(0, wsi.shape[0], args.patch_size)):
        #     if x+args.patch_size > wsi.shape[0]:
        #         continue
            
        #     for y in range(0, wsi.shape[1], args.patch_size):
        #         if y+args.patch_size > wsi.shape[1]:
        #             continue

        #         patch = wsi[x:x+args.patch_size, y:y+args.patch_size, :]
            
        #         if threshold(patch,args):
        #             im=Image.fromarray(patch)

        #             if args.save_patch_images:
        #                 im.save(Path(args.save_path / 'patches' / f'patch__{x}_{y}.png'))

        #             with torch.no_grad():
        #                 #with torch.autocast('cuda'):
                            
        #                 img_t = transform(im)

        #                 batch_t = img_t.unsqueeze(0).to(device)
        #                 features = model(batch_t)
        #                 feats.append(features)
        #                 coords=pd.concat([coords, pd.DataFrame({'scn': [scn], 'x': [x], 'y': [y]})],ignore_index=True)

        #             if args.save_tile_preview:
        #                 x1, y1 = y, x
        #                 x2, y2 = y + args.patch_size, x + args.patch_size
        #                 cv2.rectangle(wsi_copy, (x1, y1), (x2, y2), (0,0,0), thickness=4)

        # if args.save_tile_preview:
        #     preview_im=Image.fromarray(wsi_copy)
        #     preview_size = int(args.preview_size)
        #     width, height = preview_im.size
        #     aspect_ratio = height / width

        #     if aspect_ratio > 1:
        #         # Height needs to be adjusted
        #         new_height = preview_size
        #         new_width = int(preview_size / aspect_ratio)
        #     else:
        #         # Width needs to be adjusted
        #         new_width = preview_size
        #         new_height = int(preview_size * aspect_ratio)

        #     preview_im = preview_im.resize((new_width, new_height))
        #     preview_im.save(Path(args.save_path) / 'tiling_previews'/ f'{slide_name}_{scn}.png')

    # Write data to HDF5
    with h5py.File(Path(args.save_path) / 'h5_files' / f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal'/ f'{slide_name}.h5', 'w') as f:
        
        f['coords'] = coords.astype('float64')
        f['feats'] = torch.concat(feats, dim=0).cpu().numpy()
        f['args']=json.dumps(vars(args))
        f['model_name']=model_name

        print(f['coords'].shape)
        print(f['feats'].shape)
        print(time.time()-time0, " seconds for extraction")                    
        append_to_csv_file("timesp.csv",time.time()-time0)



def process_patch(wsi, x, y, args,transform,device,model,scn):
    if y + args.patch_size > wsi.shape[1]:
        return None

    patch = wsi[x:x+args.patch_size, y:y+args.patch_size, :]
    if not threshold(patch, args):
        return None

    im = Image.fromarray(patch)
    if args.save_patch_images:
        im.save(Path(args.save_path / 'patches' / f'patch__{x}_{y}.png'))

    with torch.no_grad():
        img_t = transform(im)
        batch_t = img_t.unsqueeze(0).to(device)
        features = model(batch_t)
        coord = pd.DataFrame({'scn': [scn], 'x': [x], 'y': [y]})

    return features, coord



def process_row(wsi, args, transform, device, model, scn, x, y):
    feats = []
    coords = pd.DataFrame(columns=['scn', 'x', 'y'])

    result = process_patch(wsi, x, y, args, transform, device, model, scn)
    if result:
        features, coord = result
        feats.append(features)
        coords = pd.concat([coords, coord], ignore_index=True)

    return feats, coords


# Main function
def process_wsi(wsi, args,transform,device,model,scene):
    feats = []
    coords = pd.DataFrame(columns=['scn', 'x', 'y'])

    if args.save_tile_preview:
        wsi_copy = wsi.copy()

    with ThreadPoolExecutor(max_workers=32) as executor:
        for x in tqdm(range(0, wsi.shape[0], args.patch_size)):
            if x + args.patch_size > wsi.shape[0]:
                continue

            #t=process_row(wsi, args,transform,device,model,scene,x)
            row_process_partial = partial(process_row, wsi, args, transform, device, model, scene,x)
            row_results = executor.map(row_process_partial, range(0, wsi.shape[1], args.patch_size))
            #row_results = list(executor.map(partial(process_row, wsi, args, transform, device, model, scene, x)))

            for row_feats, row_coords in row_results:
                feats.extend(row_feats)
                coords = pd.concat([coords, row_coords], ignore_index=True)

    if args.save_tile_preview:
        for index, row in coords.iterrows():
            x1, y1 = row['y'], row['x']
            x2, y2 = row['y'] + args.patch_size, row['x'] + args.patch_size
            cv2.rectangle(wsi_copy, (x1, y1), (x2, y2), (0, 0, 0), thickness=4)

    return feats, coords



if __name__=='__main__':
    args = parser.parse_args() 
    print('dd')
    main(args)
    
                                         
