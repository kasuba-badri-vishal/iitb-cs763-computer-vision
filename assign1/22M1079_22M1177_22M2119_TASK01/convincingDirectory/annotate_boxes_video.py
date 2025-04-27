"""
A Sample program to annotate bounding boxes for frames in videos
"""
import cv2
import pickle
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vidcap = cv2.VideoCapture('./../../01.mp4')
success,image = vidcap.read()
count = 0

result = []

while success:
    frame_list = [[]]
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    frame_list.append(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1
    result.append(frame_list)

with open('./../data/video/sample2.txt', 'wb') as output:
    pickle.dump([result], output, protocol=pickle.HIGHEST_PROTOCOL)
