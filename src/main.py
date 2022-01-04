import numpy as np
import cv2
from mss import mss
from PIL import Image


def main():
    bounding_box = {'top': 8, 'left': 8, 'width': 960, 'height': 540}

    sct = mss()

    while True:
        frame_raw = sct.grab(bounding_box)
        frame = np.array(frame_raw)
        detect_rectangles(frame)
        
        
        



        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        

def detect_rectangles(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

if __name__ == '__main__':
    main()


