import numpy as np
import pathlib
import random
import cv2

def SaveCompImage(fname, img1, img2):
    size_ = img1.shape
    base = np.zeros((size_[0], size_[1]*2, 3), dtype = "uint8") + 125
    base[0:size_[0], 0:size_[1]] = img1
    base[0:img2.shape[0], size_[1]:size_[1]+img2.shape[1]] = img2

    cv2.imwrite(fname, base)


def LoadImage(fname):
    return cv2.imread(str(fname))


def RandomSample(dir_, pattern = "*.png"):
    files = list(pathlib.Path(dir_).glob(pattern))
    return files[random.randint(0, len(files)-1)]