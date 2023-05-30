import cv2
import pandas as pd
import os

imgs = os.listdir("./custom_dataset/images")

choose_img = 50

img_path = os.path.join("./custom_dataset/images", imgs[choose_img], "0.jpg")
annotation_path = os.path.join("./custom_dataset/csvs", imgs[choose_img] + ".csv")

print(imgs[choose_img])

img = cv2.imread(img_path)
df = pd.read_csv(annotation_path)

h, w = img.shape[:2]
x, y = df['x'].iloc[0], df['y'].iloc[0]
x, y = int(x * w), int(y * h)

pt1 = (x-5, y-5)
pt2 = (x+5, y+5)

cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
cv2.imshow("", img)
cv2.waitKey()