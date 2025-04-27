import argparse
import xml.etree.ElementTree as ET
import cv2 as cv
import os
import numpy as np

# create an argument parser to accept the file path as an argument
parser = argparse.ArgumentParser(description='Parse an XML file.')
parser.add_argument("-f", '--filepath', type=str, help='path to the XML file', default='./../data/captured/q3/landmarks.xml')
parser.add_argument("-d", '--dir', type=str, help='path to the directory of images', default='./../data/masked/')
parser.add_argument("-c", '--color', type=str, help='color of the line (BGR format)', default='0,0,255')
# parse the command line arguments
args = parser.parse_args()

# parse the color argument
color = tuple(map(int, args.color.split(',')))

# parse the XML file
tree = ET.parse(args.filepath)
root = tree.getroot()

for image in root.iter('image'):
    filename = image.attrib["file"]
    fileloc = os.path.join(args.dir, filename)
    img = cv.imread(fileloc)
    img2 = np.copy(img)

    for box in image.findall('box'):
        # Get the value of the 'part' attribute of the first <part> element (if it exists)
        parts = box.findall(".//part")
        shape = []
        for part in parts:
            if part is not None:
                part_attribute = part.attrib
                x = int(part_attribute["x"])
                y = int(part_attribute["y"])
                shape.append([x,y])
        if(shape != []):
            points = np.array(shape)
            cv.fillPoly(img, pts=[points], color=color)
            outDirectory = "./../data/captured/q3/synthesis/fill_"+filename
            cv.imwrite(outDirectory, img)

            for i in range(len(shape)):
                if(i+1 < len(shape)):
                    img2 = cv.line(img2, shape[i], shape[i+1], color=color, thickness=2)
                else:
                    img2 = cv.line(img2, shape[i], shape[0], color=color, thickness=2)
            outDirectory = "./../data/captured/q3/synthesis/line_"+filename
            cv.imwrite(outDirectory, img2)