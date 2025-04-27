import argparse
import cv2
import os


### Canny based method to find the edges
def edge_detection(args):
    image = cv2.imread(args.img_file)
    assert image is not None, "file could not be read, check with os.path.exists()"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray,150,150)
    
    file_name = os.path.basename(args.img_file)
    cv2.imwrite("./../results/edges-" + file_name,edges)


def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection using Contours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", type=str, dest='img_file', default=None, help="path to the input data image")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    edge_detection(args)