import pandas as pd
from PIL import Image
import h5py
import slideio
import numpy as np
import cv2
from hackathon_models import get_models
from pathlib import Path
import torch
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Feature extraction')

parser.add_argument('--slide_path', help='path of slides to extract features from', default='/mnt/volume/raw_data/AIT', type=str)
parser.add_argument('--save_path', help='path to save everything', default='.', type=str)
parser.add_argument('--file_extension', help='file extension the slides are saved under, e.g. tiff', default='.czi', type=str)
parser.add_argument('--models', help='select model ctranspath, retccl, all', nargs='+', default=['retccl'], type=str)
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

#/lustre/groups/haicu/datasets/histology_data/TCGA/CRC/slides

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
            extract_features(slide, slide_name, args, model_dict,args.scene_list, device, args.BGR_to_RGB)


def extract_features(slide, slide_name,args, model_dict,scene_list,device,BGR_to_RGB): 

    model=model_dict['model']
    transform=model_dict['transforms']
    model_name=model_dict['name']

    feats=[]
    coords = pd.DataFrame({'scene' : [], 'x' : [], 'y' : []}) 

    for scn in range(slide.num_scenes):
        wsi_copy=None
        scene=slide.get_scene(scn)
        scaling=get_scaling(args,scene.resolution[0])
        wsi=scene.read_block(size=(int(scene.size[0]//scaling), int(scene.size[1]//scaling)))
        wsi=np.transpose(wsi, (1, 0, 2))
        
        if BGR_to_RGB:
            wsi = wsi[..., ::-1]
            print("! Changed BGR to RGB !")

        if args.save_tile_preview:
            wsi_copy=wsi.copy()

        for x in tqdm(range(0, wsi.shape[0], args.patch_size)):
            if x+args.patch_size > wsi.shape[0]:
                continue
            
            for y in range(0, wsi.shape[1], args.patch_size):
                if y+args.patch_size > wsi.shape[1]:
                    continue

                patch = wsi[x:x+args.patch_size, y:y+args.patch_size, :]
            
                if threshold(patch,args):
                    #extract patch
                    im=Image.fromarray(patch)

                    if args.save_patch_images:
                        im.save(Path(args.save_path / 'patches' / f'patch__{x}_{y}.png'))

                    with torch.no_grad():
                        #with torch.autocast('cuda'):
                            
                        img_t = transform(im)

                        # If the patch has a fitting shape, add it to the batch
                        batch_t = img_t.unsqueeze(0).to(device)

                        # If the batch is full or this is the last patch, process it
                        features = model(batch_t)
                        feats.append(features)
                        coords=pd.concat([coords, pd.DataFrame({'scn': [scn], 'x': [x], 'y': [y]})],ignore_index=True)

                    if args.save_tile_preview:
                        x1, y1 = y, x
                        x2, y2 = y + args.patch_size, x + args.patch_size
                        cv2.rectangle(wsi_copy, (x1, y1), (x2, y2), (0,0,0), thickness=4)

        if args.save_tile_preview:
            preview_im=Image.fromarray(wsi_copy)
            preview_size = int(args.preview_size)
            width, height = preview_im.size
            aspect_ratio = height / width

            if aspect_ratio > 1:
                # Height needs to be adjusted
                new_height = preview_size
                new_width = int(preview_size / aspect_ratio)
            else:
                # Width needs to be adjusted
                new_width = preview_size
                new_height = int(preview_size * aspect_ratio)

            preview_im = preview_im.resize((new_width, new_height))
            preview_im.save(Path(args.save_path) / 'tiling_previews'/ f'{slide_name}_{scn}.png')

    # Write data to HDF5
    with h5py.File(Path(args.save_path) / 'h5_files' / f'{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal'/ f'{slide_name}.h5', 'w') as f:
        
        f['coords'] = coords
        f['feats'] = torch.concat(feats, dim=0).cpu().numpy()

        print(f['coords'].shape)
        print(f['feats'].shape)
                            

def threshold(patch,args):
    whiteish_pixels = np.count_nonzero((patch[:, :, 0] > args.white_thresh) & (patch[:, :, 1] > args.white_thresh) & (patch[:, :, 2] > args.white_thresh))
    black_pixels = np.count_nonzero((patch[:, :, 0] <= args.black_thresh) & (patch[:, :, 1] <= args.black_thresh) & (patch[:, :, 2] <= args.black_thresh))
    
    # Compute the ratio of foreground pixels to total pixels in the patch
    invalid_ratio = (whiteish_pixels + black_pixels) / (patch.shape[0] * patch.shape[1])

    if invalid_ratio<=args.invalid_ratio_thresh:

        edge  = cv2.Canny(patch, 40, 100) 
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


def get_driver(extension_name):
    if extension_name in['.tiff','.jpg','.jpeg','.png','.tif']:
        return 'GDAL'
    elif extension_name=='':
        return 'DICOM'
    else:
        return extension_name.replace('.','').upper()
    
def get_scaling(args,mpp_resolution_slide):
    if args.downscaling_factor>0:
        return args.downscaling_factor
    else:
        return args.resolution_in_mpp/(mpp_resolution_slide*1e06)
    
if __name__=='__main__':
    args = parser.parse_args() 
    print('dd')
    main(args)
    
                                         
