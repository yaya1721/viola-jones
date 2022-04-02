#modified from https://github.com/salvacarrion/viola-jones

import time
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2 as cv
import multiprocessing

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # self.x1 = int(x - 1)
        # self.y1 = int(y - 1)
        # self.x2 = int(x + width - 1)
        # self.y2 = int(y + height - 1)

    def compute_region(self, ii, scale=1.0): #integral image
        # D(all) - C(left) - B(top) + A(corner)

        # x1 = self.x
        # y1 = self.y
        # x2 = x1 + self.width - 1
        # y2 = y1 + self.height - 1

        x1 = int(self.x * scale)
        y1 = int(self.y * scale)
        x2 = x1 + int(self.width * scale) - 1
        y2 = y1 + int(self.height * scale) - 1
        
        S = int(ii[x2, y2])
        if x1 > 0: S -= int(ii[x1-1, y2])
        if y1 > 0: S -= int(ii[x2, y1-1])
        if x1 > 0 and y1 > 0: S += int(ii[x1 - 1, y1 - 1])
        return S  # Due to the use of substraction with unsigned values


class HaarFeature:
    def __init__(self, positive_regions, negative_regions):
        self.positive_regions = positive_regions  # White
        self.negative_regions = negative_regions  # Black

    def compute_value(self, ii, scale=1.0):
        """
        Compute the value of a feature(x,y,w,h) at the integral image
        """
        sum_pos = sum([rect.compute_region(ii, scale) for rect in self.positive_regions])
        sum_neg = sum([rect.compute_region(ii, scale) for rect in self.negative_regions])
        return sum_neg - sum_pos


def integral_image(img):
    s = np.zeros(img.shape, dtype=np.uint32)
    ii = np.zeros(img.shape, dtype=np.uint32)

    for x in range(0, 19):
        for y in range(0, 19):
            s[y][x] = s[y - 1][x] + img[y][x] if y - 1 >= 0 else img[y][x]
            ii[y][x] = ii[y][x - 1] + s[y][x] if x - 1 >= 0 else s[y][x]
    return ii

def build_features(img_w = 19, img_h = 19, shift=1, scale_factor=1.25, min_w=4, min_h=4):
    """
    Generate values from Haar features
    White rectangles substract from black ones
    """
    features = []  # [Tuple(positive regions, negative regions),...]

    # Scale feature window
    for w_width in range(min_w, img_w + 1):
        for w_height in range(min_h, img_h + 1):

            # Walk through all the image
            x = 0
            while x + w_width < img_w:
                y = 0
                while y + w_height < img_h:

                    # Possible Haar regions
                    immediate = RectangleRegion(x, y, w_width, w_height)  # |X|
                    right = RectangleRegion(x + w_width, y, w_width, w_height)  # | |X|
                    right_2 = RectangleRegion(x + w_width * 2, y, w_width, w_height)  # | | |X|
                    bottom = RectangleRegion(x, y + w_height, w_width, w_height)  # | |/|X|
                    #bottom_2 = RectangleRegion(x, y + w_height * 2, w_width, w_height)  # | |/| |/|X|
                    bottom_right = RectangleRegion(x + w_width, y + w_height, w_width, w_height)  # | |/| |X|

                    # [Haar] 2 rectagles 
                    # Horizontal (w-b)
                    if x + w_width * 2 < img_w:
                        features.append(HaarFeature([immediate], [right]))
                    # Vertical (w-b)
                    if y + w_height * 2 < img_h:
                        features.append(HaarFeature([bottom], [immediate]))

                    # [Haar] 3 rectagles 
                    # Horizontal (w-b-w)
                    if x + w_width * 3 < img_w:
                        features.append(HaarFeature([immediate, right_2], [right]))
                    # # Vertical (w-b-w)
                    # if y + w_height * 3 < img_h:
                    #     features.append(HaarFeature([immediate, bottom_2], [bottom]))

                    # [Haar] 4 rectagles 

                    if x + w_width * 2 < img_w and y + w_height * 2 < img_h:
                        features.append(HaarFeature([immediate, bottom_right], [bottom, right]))

                    y += shift
                x += shift
    return features  # np.array(features)


def apply_features(X_ii, features):
    """
    Apply build features (regions) to all the training data (integral images)
    """

    X = np.zeros((len(features), len(X_ii)), dtype=np.int32)
    # 'y' will be kept as it is => f0=([...], y); f1=([...], y),...

    #bar = Bar('Processing features', max=len(features), suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
    for j, feature in enumerate(features):
    # for j, feature in enumerate(features):
    #     if (j + 1) % 1000 == 0 and j != 0:
    #         print("Applying features... ({}/{})".format(j + 1, len(features)))

        # Compute the value of feature 'j' for each image in the training set (Input of the classifier_j)
        X[j] = list(map(lambda ii: feature.compute_value(ii), X_ii))
    #bar.finish()

    return X


def draw_haar_feature(np_img, haar_feature):
    pil_img = Image.fromarray(np_img).convert("RGBA")

    draw = ImageDraw.Draw(pil_img)
    for rect in haar_feature.positive_regions:
        x1, y1, x2, y2 = rect.x, rect.y, rect.x + rect.width - 1, rect.y + rect.height - 1
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, 255))

    for rect in haar_feature.negative_regions:
        x1, y1, x2, y2 = rect.x, rect.y, rect.x + rect.width - 1, rect.y + rect.height - 1
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 255))

    return pil_img