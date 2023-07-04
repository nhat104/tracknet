PATH_TO_DATASET="/content/tracknet/example_datasets/images_dataset"
session_name="example-dataset-heatmapv3-10000epochs"
weight_path="/content/best-10000epochs.pth"
!python mytrain.py --dataset $PATH_TO_DATASET --device cuda --epochs 1 --image_size 360 640 --train_size 0.8 --sequence_length 1 --session_name $session_name --batch_size 8 --loss mse --weights $weight_path