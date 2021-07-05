import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    def __init__(
            self,
            dataset: xr.Dataset,
            symbol: str,
            num_observations: int = 50,
            num_steps_ahead: int = 10,
            transform=None
    ):
        self.dataset = dataset
        self.symbol = symbol
        self.num_observations = num_observations
        self.num_steps_ahead = num_steps_ahead
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label