from HumanSegmentation import BodyDetector
from bgRemover import *
import cv2
from Utils import *

detector = BodyDetector("model_weights/bgrm-bh.pth")

fname = RandomSample("background folder", '*')
bg = LoadImage(fname)

fname = RandomSample("image folder")
img = LoadImage(fname)
img_resize = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)

mask = detector.DetectBody(img_resize)

res = ReplaceBG(img, mask, bg)

SaveCompImage(f"{fname.stem}.jpg", img, res)