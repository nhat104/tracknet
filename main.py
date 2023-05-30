import json
import cv2
import os
import shutil
import argparse

def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw_path", required=True, type=str)
  parser.add_argument("--destination_path", required=True, type=str)
  
  opt = parser.parse_args()
  return opt

if __name__ == "__main__":
  opt = vars(parse_opt())
  
  raw_path = opt['raw_path']
  destination_path = opt["destination_path"]
  
  data = json.load(open(os.path.join(raw_path, "_annotations.coco.json"), "r"))
  images = data["images"]
  annotations = data["annotations"]

  annotation_base_dir = os.path.join(destination_path, "csvs")

  for annotation in annotations:
    image_id = annotation['image_id']
    image = images[image_id]
    folder_name = os.path.join(destination_path, "images", image["file_name"].split(".")[0])
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
      os.path.join(raw_path, image["file_name"]),
      os.path.join(folder_name, "0.jpg")
    )