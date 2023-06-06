from utils.dataset import TrackNetDataset
from utils.dataloader import TrackNetDataloader

csv_path = "example_datasets/images_dataset/csvs/VolleyballShortClip.csv"
dataset = TrackNetDataset(csv_path)
dataloader = TrackNetDataloader(dataset, batch_size=64, shuffle=True)