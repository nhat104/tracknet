import cv2
import numpy as np
import glob
import os

frameSize = (720.0, 1280.0)

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'H264'), 25, (1280, 720))

for filename in glob.glob('./test_video/*.jpg'):
    img = cv2.imread(os.path.join(filename))
    out.write(img)

out.release()