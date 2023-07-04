from argparse import ArgumentParser
from TrackNet import TrackNet
import torchvision
import torch
import cv2 as cv
import os
import numpy as np


def get_ball_position(img, opt, original_img_=None):
    ret, thresh = cv.threshold(img, opt.brightness_thresh, 1, 0)
    thresh = cv.convertScaleAbs(thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key = cv.contourArea)

        if original_img_ is not None:
            # the contours are drawn here
            cv.drawContours(original_img_, [c], -1, 255, 3)

        x,y,w,h = cv.boundingRect(c)
        return x, y, w, h


def parse_opt():
    parser = ArgumentParser()
#    parser.add_argument('video', type=str, default='video.mp4', help='Path to video.')
    parser.add_argument('--save_path', type=str, default='predicted.mp4', help='Path to result video.')
    parser.add_argument('--weights', type=str, default='weights', help='Path to trained model weights.')
    parser.add_argument('--sequence_length', type=int, default=3, help='Length of the images sequence used as X.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[720, 1280], help='Size of the images used for training (y, x).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps).')
    parser.add_argument('--one_output_frame', action='store_true', help='Demand only one output frame instead of three.')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images instead of RGB.')
    parser.add_argument('--visualize', action='store_true', help='Display the predictions in real time.')
    parser.add_argument('--waitBetweenFrames', type=int, default=100, help='Wait time in milliseconds between showing frames predicted in one forward pass.')
    parser.add_argument('--brightness_thresh', type=float, default=0.7, help='Result heatmap pixel brightness threshold')
    parser.add_argument('--image_path', type=str, required=True, help='Image path for predict')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    img_path = vars(opt)['image_path']
    
    opt.dropout = 0
    device = torch.device(opt.device)
    model = TrackNet(opt).to(device)
    # model.load(opt.weights, device = opt.device)
    model.load_state_dict(torch.load(opt.weights, map_location=torch.device('cpu')))
    model.eval()

    img = cv.imread(img_path)
    
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    image = torchvision.transforms.ToTensor()(rgb_image)
    image = torchvision.transforms.Resize(size=opt.image_size, antialias=True)(image)
    image = image.type(torch.float32)
    # image *= 1 / 255
    image = image.unsqueeze(0)
    
    pred = model(image)
    pred_frame = pred[0, 0]
    pred_frame = pred_frame.detach().numpy()
    
    heatmap = (pred_frame * 255).astype(np.uint8)
    cv.imwrite("./output/heatmap.jpg", heatmap)
    
    y, x = np.where(pred_frame == np.max(pred_frame))
    x, y = x[0], y[0]
    
    h, w = img.shape[:2]
    center = (int(x / opt.image_size[1] * w), int(y / opt.image_size[0] * h))
    print(center)
    cv.circle(img, center, 5, (0, 255, 0), 2)
    cv.imshow("", img)
    cv.waitKey()