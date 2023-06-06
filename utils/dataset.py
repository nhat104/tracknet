import os
from typing import Any
from torch.utils.data import Dataset

import torch
from torchvision.io import read_image
import torchvision
from argparse import Namespace

import pandas as pd
import numpy as np

class TrackNetDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, opt: Namespace) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_frame = pd.read_csv(csv_file)
        self.image_size = opt.image_size
        
    def __getitem__(self, index) -> Any:
        x = self.image_frame['x'].iloc[index]
        y = self.image_frame['y'].iloc[index]
        img_path = os.path.join(self.root_dir, f"{index}.png")
        
        # read image and scale to [0, 1]
        image = read_image(img_path)
        image = torchvision.transforms.functional.resize(image, self.image_size, antialias=True)
        image = image.type(torch.float32)

        image = image / 255
        
        heatmap = self.generate_heatmap(x, y)
        return image, heatmap
    
    def __len__(self) -> int:
        return self.image_frame.shape[0]
    
    def generate_heatmap(self, center_x, center_y):
        """Generate heatmap based on original paper

        Args:
            center_x (int): _description_
            center_y (int): _description_
            width (int): _description_
            height (int): _description_

        Returns:
            _type_: _description_
        """
        
        x0 = self.image_size[1]*center_x
        y0 = self.image_size[0]*center_y
        
        x = [np.arange(0, self.image_size[1], 1, float)] * self.image_size[0]
        x = np.stack(x)
        y = [np.arange(0, self.image_size[0], 1, float)] * self.image_size[1]
        y = np.stack(y).T

        sigma = 10

        G = (x - x0) ** 2 + (y - y0) ** 2
        G = -G / (2 * sigma)
        G = np.exp(G)
        
        G = torch.Tensor(G)
        # upper_G = G > 0.5
        # G[upper_G] = 1
        # G[~upper_G] = 0
        return G.type(torch.float32)