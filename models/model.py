import torch
import torch.nn as nn
from torchvision import transforms

from torchvision.models import resnet
from models.ctran import ctranspath
from models.kimianet import load_kimianet
from models.resnet_retccl import resnet50 as retccl_res50
from models.simsalabim import ResNetSimCLR
from models.sam import build_sam_vit_h,build_sam_vit_b,build_sam_vit_l
from models.imagebind import imagebind_huge
from transformers import Data2VecVisionModel, BeitFeatureExtractor

# RetCCL can be downloaded here: https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL?usp=sharing
# kimianet download: https://kimialab.uwaterloo.ca/kimia/?smd_process_download=1&download_id=4216
RETCCL_PATH = '/home/ubuntu/run/retccl.pth'
CTRANSPATH_PATH = '/lustre/groups/shared/users/peng_marr/pretrained_models/ctranspath.pth'
KIMIANET_PATH = '/mnt/volume/models/KimiaNetPyTorchWeights.pth'
SIMCLR_LUNG_PATH= '/mnt/volume/models/rushinssimclr.pth' 
SAM_VIT_H_PATH='/mnt/ceph_vol/models/sam_vit_h_4b8939.pth'
SAM_VIT_L_PATH="/mnt/ceph_vol/models/sam_vit_l_0b3195.pth"
SAM_VIT_B_PATH="/mnt/ceph_vol/models/sam_vit_b_01ec64.pth"

def get_models(modelnames):
    models = []
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    for modelname in modelnames:
        if modelname.lower() == 'retccl':
            model = get_retCCL()
        elif modelname.lower() == 'ctranspath':
            model = get_ctranspath()
        elif modelname.lower() == 'resnet50':
            model = get_res50()
        elif modelname.lower() == "kimianet":
            model = get_kimianet()
        elif modelname.lower() == "simclr_lung":
            model = get_simclr_lung()
        elif modelname.lower()=="sam_vit_h":
            model=get_sam_vit_h()
        elif modelname.lower()=="sam_vit_b":
            model=get_sam_vit_b()
        elif modelname.lower()=="sam_vit_l":
            model=get_sam_vit_l()
        elif modelname.lower() in ['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14']:
            model=torch.hub.load('facebookresearch/dinov2', modelname.lower())
        elif modelname.lower()=="imagebind":
            model=get_imagebind()
        elif modelname.lower()=='beit_fb':
            model = BeitModel(device)
        """
        # torch.compile does not work with DataParallel
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        """
        model.to(device)
        #model = torch.compile(model)
        model.eval()
        transforms = get_transforms(modelname)
        #models.append({'name': modelname, 'model': torch.compile(
        #    model.to(device)), 'transforms': transforms})
        models.append({'name': modelname, 'model': 
            model.to(device), 'transforms': transforms})
    return models


def get_sam_vit_h():
    return build_sam_vit_h(SAM_VIT_H_PATH)

def get_sam_vit_l():
    return build_sam_vit_l(SAM_VIT_L_PATH)

def get_sam_vit_b():
    return build_sam_vit_b(SAM_VIT_B_PATH)

def get_retCCL():
    model = retccl_res50(num_classes=128, mlp=False,
                         two_branch=False, normlinear=True)
    pretext_model = torch.load(RETCCL_PATH)
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model

def get_ctranspath():
    model = ctranspath()
    model.head = nn.Identity()
    pretrained = torch.load(CTRANSPATH_PATH)
    model.load_state_dict(pretrained['model'], strict=True)
    return model


def get_kimianet():
    return load_kimianet(KIMIANET_PATH)

def get_simclr_lung():
    model=ResNetSimCLR()
    pretrained = torch.load(SIMCLR_LUNG_PATH)
    model.load_state_dict(pretrained,strict=False)
    return model

def get_res50():

    model = resnet.resnet50(weights='ResNet50_Weights.DEFAULT')

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)
        
    #delete last res block as this has been shown to work better
    model = nn.Sequential(*list(model.children())
                          [:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

    return model

def get_imagebind(pretrained=True):
    model = imagebind_huge(pretrained=pretrained)
    return model

def multiply_by_255(img):
    return img * 255

def get_transforms(model_name):
    # from imagenet, leave as is
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ['ctranspath', 'resnet50',"simclr_lung", 'beit_fb']:
        resolution = 224
    elif model_name.lower() == 'retccl':
        resolution = 256
    elif model_name.lower() == 'kimianet':
        resolution = 1000
    elif model_name.lower() == 'imagebind':
        resolution = 224
        mean=(0.48145466, 0.4578275, 0.40821073)
        std=(0.26862954, 0.26130258, 0.27577711)
    # change later to correct value
    elif model_name.lower() in ['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14']:
        resolution = 252
    elif "sam" in model_name.lower():
        resolution = 1024
        mean=(123.675, 116.28, 103.53)
        std=(58.395, 57.12, 57.375)
    else:
        raise ValueError('Model name not found')

    transforms_list = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if 'beit_fb' in model_name.lower():
        transforms_list = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ]
    
    elif "sam" in model_name.lower():
        # multiply image by 255 for "sam" model
        transforms_list = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Lambda(multiply_by_255),
        transforms.Normalize(mean=mean, std=std),
    ]

    preprocess_transforms = transforms.Compose(transforms_list)
    return preprocess_transforms


class BeitModel(torch.nn.Module):
    def __init__(self, device, pretrained_model='facebook/data2vec-vision-base', image_size=224, patch_size=16):
        super(BeitModel, self).__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(pretrained_model)
        self.image_size = image_size
        self.patch_size = patch_size
        self.model =  Data2VecVisionModel.from_pretrained(pretrained_model)
        self.device=device
        self.avg_pooling=nn.AdaptiveAvgPool1d((1))

    def forward(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs=inputs['pixel_values'].to(self.device)
        outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = outputs.hidden_states

        # The provided code was taking only the 13th layer, I kept that behaviour
        features = encoder_hidden_states[12][:,1:,:].permute(0,2,1)
        features=self.avg_pooling(features)

        return features.squeeze()
    

if __name__ == '__main__':
    get_models(['resnet50'])