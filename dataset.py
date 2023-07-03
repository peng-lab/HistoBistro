from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class SlideDataset(Dataset):
    def __init__(self, slide, coordinates: pd.DataFrame, patch_size: int, transform:list =None):
        super(SlideDataset, self).__init__()
        self.slide=slide
        self.coordinates=coordinates
        self.patch_size=patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        print(idx)
        x=self.coordinates.iloc[idx]['x']
        y=self.coordinates.iloc[idx]['y']

        patch = self.slide[x:x+self.patch_size, y:y+self.patch_size, :]
        img = Image.fromarray(patch)  # Convert image to RGB

        if self.transform:
            img = self.transform(img)

        return img


