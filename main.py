import json
import cv2
import os
import shutil

data = json.load(open("images/train/_annotations.coco.json", "r"))
images = data["images"]
annotations = data["annotations"]

train_annotation = "train_annotation.txt"
file_writer = open(train_annotation, "w")

annotation_base_dir = "./custom_dataset/csvs"

for image, annotation in zip(images, annotations):
  folder_name = os.path.join("./custom_dataset/images", image["file_name"].split(".")[0])
  writer = open(os.path.join(annotation_base_dir, image["file_name"].split(".")[0] + ".csv"), "w")
  point_center = (int(annotation["bbox"][0] + annotation["bbox"][2] / 2), int(annotation["bbox"][1] + annotation["bbox"][3] / 2))
  file_writer.write(f"{image['file_name']}\t{point_center[0]}\t{point_center[1]}\n")
  writer.write("frame_num,x,y,visible\n")
  writer.write(f"0,{point_center[0]/720},{point_center[1]/1280},0\n")
  writer.close()

  if not os.path.exists(folder_name):
    os.makedirs(folder_name)

  shutil.copy(
    os.path.join("images/train", image["file_name"]),
    os.path.join(folder_name, "0.jpg")
  )

file_writer.close()
    
# image_path = "train/" + image["file_name"]
# img = cv2.imread(image_path)
# img[point_center] = (0, 0, 0)
# img = cv2.circle(img, point_center, 5, (0, 0, 255), -1)
# cv2.imshow("img", img)
# cv2.waitKey(0)

