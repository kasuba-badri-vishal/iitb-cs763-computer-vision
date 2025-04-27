import cv2
import os

count = 1

for i in range(10,66):
    img = cv2.imread('./../data/masked/images/'+str(i)+'.jpg')
    print("<image file='"+str(i)+".jpg' width='"+str(img.shape[1])+"' height='"+str(img.shape[0])+"'>")

# img = cv2.imread('./../data/masked/images1/masked04.jpg')
# print(img.shape)

# img = cv2.imread('./../data/masked/images1/masked05.jpg')
# print(img.shape)

# img = cv2.imread('./../data/masked/images1/masked06.jpg')
# print(img.shape)