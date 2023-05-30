from typing import Any
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)