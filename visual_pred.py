from argparse import ArgumentParser
from TrackNet import TrackNet
import torchvision
import torch
import cv2 as cv
import os
import numpy as np


def generate_heatmap_2(opt, center_x, center_y, width, height):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    source: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
    """
    x0 = opt.image_size[1]*center_x
    y0 = opt.image_size[0]*center_y
    
    x = [np.arange(0, opt.image_size[1], 1, float)] * opt.image_size[0]
    x = np.stack(x)
    y = [np.arange(0, opt.image_size[0], 1, float)] * opt.image_size[1]
    y = np.stack(y).T

    sigma = 10

    G = (x - x0) ** 2 + (y - y0) ** 2
    G = -G / (2 * sigma)
    G = np.exp(G)
    
    upper_G = G > 0.5
    G[upper_G] = 1
    G[~upper_G] = 0
    return G


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
    # img_path = vars(opt)['image_path']
    img_path = opt.image_path
    
    opt.dropout = 0
    device = torch.device(opt.device)
    model = TrackNet(opt).to(device)
    # model.load(opt.weights, device = opt.device)
    # model.eval()
    model.load_state_dict(torch.load(opt.weights))
    model = model.to(device)
    model.eval()
    
    img = torchvision.io.read_image(img_path)
    img = torchvision.transforms.Grayscale()(img)
    img = torchvision.transforms.functional.resize(img, opt.image_size, antialias=True)
    img = img.type(torch.float32)
    img *= 1 / 255
    
    img = img.unsqueeze(0)
    
    img = img.to(device)
    
    print(img.shape)

    pred = model(img)
    pred_heatmap = pred[0, 0]
    
    print(pred_heatmap)
    print(pred_heatmap.max(), pred_heatmap.min())
    print(get_ball_position(pred_heatmap.detach().cpu().numpy(), opt, None))
    
    heatmap_numpy = pred_heatmap.detach().cpu().numpy() * 255
    heatmap_numpy = heatmap_numpy.astype("uint")
    print(heatmap_numpy)
    print(np.where(heatmap_numpy == heatmap_numpy.max()))
    cv.imwrite("/content/output/heatmap.png", heatmap_numpy)
    
    gt_heatmap = generate_heatmap_2(opt, 0.43828125, 0.6361111111111111, 5, 5)
    print(gt_heatmap.min(), gt_heatmap.max())
    print(get_ball_position(gt_heatmap, opt, None))
    print(np.where(gt_heatmap == gt_heatmap.max()))
    gt_heatmap = gt_heatmap * 255
    gt_heatmap = gt_heatmap.astype("uint")
    cv.imwrite("/content/output/gt.png", gt_heatmap)
    