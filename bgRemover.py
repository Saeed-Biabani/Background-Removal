import cv2


def ReplaceBG(img, mask, bg):
    mask = cv2.resize(mask, (list(reversed(img.shape))[1:]), interpolation = cv2.INTER_AREA)
    res = cv2.bitwise_and(img, img, mask = mask)
    bg = cv2.resize(bg, (list(reversed(img.shape))[1:]), interpolation = cv2.INTER_AREA)
    resb = cv2.bitwise_or(bg, bg, mask = cv2.bitwise_not(mask))
    return cv2.add(resb, res)


def RemoveBG(img, mask):
    mask = cv2.resize(mask, (list(reversed(img.shape))[1:]), interpolation = cv2.INTER_AREA)
    return cv2.bitwise_and(img, img, mask = mask)