import torch
import torch.nn as nn
import torchvision

class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		return  out_1

def load_kimianet(model_path):
	model = torchvision.models.densenet121(weights=None)
	for param in model.parameters():
		param.requires_grad = False
	model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
	num_ftrs = model.classifier.in_features
	model_final = fully_connected(model.features, num_ftrs, 30)
	model_final = nn.DataParallel(model_final)

	model_final.load_state_dict(torch.load(model_path),strict=False) #getting rid of final layer for classification

	return model