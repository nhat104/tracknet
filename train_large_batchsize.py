from dataset import TrackNetDataset
from tracknet.TrackNet import TrackNet
from argparse import Namespace
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

opt = Namespace()
opt.image_size = [360, 640]
opt.grayscale = False
opt.sequence_length = 1
opt.dropout = 0
opt.one_output_frame = False

full_dataset = TrackNetDataset(csv_file="/content/tracknet/example_datasets/images_dataset/csvs/VolleyballShortClip.csv", root_dir="/content/tracknet/example_datasets/images_dataset/images/VolleyballShortClip", opt=opt)
loader = DataLoader(full_dataset, batch_size=8, shuffle=True)
tracknet = TrackNet(opt)
device = "cuda:0"
tracknet = tracknet.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(tracknet.parameters())
print(torch.cuda.max_memory_allocated())
for epoch in tqdm(range(100)):
    for batch_data in loader:
        optimizer.zero_grad()
        X, y = batch_data
        X = X.to(device)
        y = y.to(device)
        y_pred = tracknet(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        tqdm.write(f"{loss.detach().item()}")


