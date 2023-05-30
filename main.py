import json
import cv2
import os
import shutil

data = json.load(open("images/valid/_annotations.coco.json", "r"))
images = data["images"]
annotations = data["annotations"]

annotation_base_dir = "./custom_dataset_1/csvs"

for image, annotation in zip(images, annotations):
  folder_name = os.path.join("./custom_dataset_1/images", image["file_name"].split(".")[0])
  writer = open(os.path.join(annotation_base_dir, image["file_name"].split(".")[0] + ".csv"), "w")
  
  point_center = (int(annotation["bbox"][0] + annotation["bbox"][2] / 2), int(annotation["bbox"][1] + annotation["bbox"][3] / 2))
  writer.write("frame_num,x,y,visible\n")
  
  # scale center to range (0, 1)
  width = image['width']
  height = image['height']
  
  x = point_center[0] / width
  y = point_center[1] / height
  
  writer.write(f"0,{x},{y},0\n")
  writer.close()

  if not os.path.exists(folder_name):
    os.makedirs(folder_name)

  shutil.copy(
    os.path.join("images/valid", image["file_name"]),
    os.path.join(folder_name, "0.jpg")
  )