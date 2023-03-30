import xml.etree.ElementTree as ET
import numpy as np
import cv2
import csv

def bgr_format(xml_string): #check if BGR or RGB
    root = ET.fromstring(xml_string)
    pixel_type_elem = root.findall(".//PixelType")
    return 'bgr' in pixel_type_elem[0].text.lower() if pixel_type_elem is not None else False


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
    
    

def append_to_csv_file(filename, number):
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([number])
