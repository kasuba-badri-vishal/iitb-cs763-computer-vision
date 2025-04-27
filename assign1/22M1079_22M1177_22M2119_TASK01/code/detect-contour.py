import argparse
import cv2
import os

#### Contour based method to find tallest person
def contour_detection(args):
    
    image = cv2.imread(args.img_file)
    assert image is not None, "file could not be read, check with os.path.exists()"
    
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    height, width, _ = image.shape
    area_not_include = (height-1)*(width-1)
    contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if(area>max_area and area<area_not_include):
            max_area = area
            contour = c

    cv2.drawContours(image, contour, -1, (0,255,0), 3)
    
    file_name = os.path.basename(args.img_file)
    cv2.imwrite("./../results/" + file_name,image)


def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection using Contours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", type=str, dest='img_file', default=None, help="path to the input data image")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    contour_detection(args)