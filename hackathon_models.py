import feature_extractors.ResNet as ResNet
from feature_extractors.ccl import CCL
import torch.nn as nn
import torch

#import requests
from pathlib import Path
from torchvision import transforms
from feature_extractors.ctran import ctranspath
#from PIL import Image
import torchvision.models as models


# RetCCL can be downloaded here: https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL?usp=sharing
RETCCL_PATH = '/lustre/groups/shared/users/peng_marr/pretrained_models/RetCCL'
CTRANSPATH_PATH= '/lustre/groups/shared/users/peng_marr/pretrained_models/CTransPath/ctranspath.pth'

def get_models(modelnames):
    models=[]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for modelname in modelnames:
        if modelname.lower() == 'retccl':
            model= get_retCCL()
        elif modelname.lower() == 'ctranspath':
            model= get_ctranspath()
        elif modelname.lower()=='resnet50':
            model=get_res50()
        model.to(device)
        model=torch.compile(model)
        model.eval()
        transforms=get_transforms(modelname)
        models.append({'name':modelname, 'model':torch.compile(model.to(device)), 'transforms':transforms})
    return models


def get_retCCL():
    backbone = ResNet.resnet50
    model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()
    pretext_model = torch.load(RETCCL_PATH)
    model.load_state_dict(pretext_model, strict=True)
    model.encoder_q.fc = nn.Identity()
    model.encoder_q.instDis = nn.Identity()
    model.encoder_q.groupDis = nn.Identity()
    model.eval()
    return model

def get_ctranspath():
    model=ctranspath()
    model.head = nn.Identity()
    pretrained = torch.load(CTRANSPATH_PATH)
    model.load_state_dict(pretrained['model'], strict=True)
    model=torch.compile(model)
    model.eval()
    return model
    
def kimianet():
    pass

def get_res50():

    model = models.resnet50(pretrained=True)
    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)
    
    model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1,1)), Reshape())
  
    return model

def get_transforms(model_name):
    #from imagenet, leave as is
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ['ctranspath','resnet50']:
        resolution=224
    elif model_name.lower()=='retccl':
        resolution=256
    else: 
        raise ValueError('Model name not found')
    
    preprocess_transforms = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    return preprocess_transforms

if __name__=='__main__':
    get_models(['resnet50'])


