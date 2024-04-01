import numpy as np
import cv2 as cv
from rembg import remove
from PIL import Image

cap = cv.VideoCapture(0)

#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
#fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

while (1):
    ret, frame = cap.read()

    #fgmask = fgbg.apply(frame)
    #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    output = remove(frame)
    cv.imshow('frame', output)
    if cv.waitKey(1) == 13: #enter
        break

cap.release()
cv.destroyAllWindows()