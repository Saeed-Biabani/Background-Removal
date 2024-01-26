from HumanSegmentation import BodyDetector
from bgRemover import *
from Utils import *
import cv2


detector = BodyDetector("model_weights/bgrm-bh.pth")

fname = RandomSample("bgfolder", '*')
bg = LoadImage(fname)

cap = cv2.VideoCapture(0)

while True:
    re, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    
    mask = detector.DetectBody(img)
    
    if bg.shape != frame.shape:
        bg = cv2.resize(bg, list(reversed(frame.shape))[1:], interpolation = cv2.INTER_AREA)
    
    res = ReplaceBG(frame, mask, bg)
    
    cv2.imshow("frame", frame)
    cv2.imshow("mask", res)

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()