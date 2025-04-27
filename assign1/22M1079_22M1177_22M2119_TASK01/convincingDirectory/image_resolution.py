import cv2
import os

input_dir = './../data/captured/'

for file in sorted(os.listdir(input_dir)):
    image = cv2.imread(input_dir+file)
    print(file,image.shape)