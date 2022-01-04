import numpy as np
import cv2
from mss import mss
from PIL import Image


def main():

    # Load the image
    img = cv2.imread('sample.png')

    # convert to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('egdes', gray)
    ret, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)


    # Find the contours
    contours, hierarchy = cv2.findContours(thresh1,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img,
                      (x, y), (x + w, y + h),
                      (0, 255, 0),
                      2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


