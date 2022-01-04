import numpy as np
import cv2
from mss import mss
from PIL import Image


def main():
    bounding_box = {'top': 8, 'left': 8, 'width': 960, 'height': 540}

    sct = mss()

    while True:
        frame = sct.grab(bounding_box)
        cv2.imshow('screen', np.array(frame))
        
        



        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()


