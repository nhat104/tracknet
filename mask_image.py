import cv2
import json
import os

data = json.load(open("images/train/_annotations.coco.json", "r"))
images = data["images"]
annotations = data["annotations"]

train_annotation = "train_annotation.txt"
annotation_base_dir = "./custom_dataset_1/csvs"

images = images[0]
annotations = annotations[0]

# for image, annotation in zip(images, annotations):
point_center = (int(annotations["bbox"][0] + annotations["bbox"][2] / 2), int(annotations["bbox"][1] + annotations["bbox"][3] / 2))
width = images['width']
height = images['height']

pt1 = (point_center[0] - 5, point_center[1] - 5)
pt2 = (point_center[0] + 5, point_center[1] + 5)

img = cv2.imread(os.path.join("images/train", images['file_name']))
cv2.rectangle(img, pt1, pt2, (255, 0, 0), thickness=2)
# cv2.imshow("", img)
# cv2.waitKey()

x = point_center[0] / width
y = point_center[1] / height