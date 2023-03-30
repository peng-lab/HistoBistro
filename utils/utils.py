import xml.etree.ElementTree as ET

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
    