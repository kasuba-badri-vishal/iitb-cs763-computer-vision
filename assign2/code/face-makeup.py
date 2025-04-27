import cv2
import numpy as np
import face_recognition
import os
import argparse 

def face_makeup_main():
    #Defining parser
    parser = argparse.ArgumentParser(prog="face-makeup",
		description="Beautifies the face",
		epilog="Thank you for using !!",
		argument_default=argparse.SUPPRESS,
		allow_abbrev=False,
		fromfile_prefix_chars="@")
    #Arguments
    parser.add_argument("path_input")
    parser.add_argument("path_output")
    parser.add_argument("value")


    #Flags
    parser.add_argument("-i", "--input", action="store_true", required=True, help="Path to input")
    parser.add_argument("-o", "--output", action="store_true", required=True, help="Path to save results")
    parser.add_argument("-t", "--type", action="store_true", required=True, help="File application")

    args = parser.parse_args()
    #print(f'NameSpace : {args}')

    directory = os.path.split(args.path_output)
    if( not os.path.exists(directory[0])):
         os.makedirs(directory[0])
    
    #input path
    path = args.path_input
    # Read image with CV and a copy for overlaying
    cv2_image = cv2.imread(path)
    overlay = cv2.imread(path)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(cv2_image)
    #parser type value
    para = args.value
    if para == "1" :
        #MAKE LINES
        color = (255,255,255)
        thickness = 1
        for face_landmarks in face_landmarks_list:
            for part in face_landmarks.keys():
                for point_index in range(len(face_landmarks[part])-1):
                    cv2.line(cv2_image, face_landmarks[part][point_index], face_landmarks[part][point_index+1], color, thickness)
        cv2.imwrite(args.path_output, cv2_image)

    elif para == "2":
        for face_landmarks in face_landmarks_list:
            # Making the eyebrows dark
            cv2.fillPoly(cv2_image, pts=[np.array(face_landmarks['left_eyebrow'])], color=(39, 54, 68))
            cv2.fillPoly(cv2_image, pts=[np.array(face_landmarks['right_eyebrow'])], color=(39, 54, 68))
            part = 'right_eyebrow'
            for point_index in range(len(face_landmarks[part])-1):
                    cv2.line(cv2_image, face_landmarks[part][point_index], face_landmarks[part][point_index+1], color=(39, 54, 68), thickness=4)
            part = 'left_eyebrow'
            for point_index in range(len(face_landmarks[part])-1):
                    cv2.line(cv2_image, face_landmarks[part][point_index], face_landmarks[part][point_index+1], color=(39, 54, 68), thickness=4)

            # Gloss the lips
            cv2.fillPoly(cv2_image, pts=[np.array(face_landmarks['top_lip'])], color=(0, 0, 150))
            cv2.fillPoly(cv2_image, pts=[np.array(face_landmarks['bottom_lip'])], color=(0, 0, 150))

            # Apply some eyeliner
            part = 'left_eye'
            for point_index in range(len(face_landmarks[part])-1):
                    cv2.line(cv2_image, face_landmarks[part][point_index], face_landmarks[part][point_index+1], color=(0, 0, 0), thickness=4)
            cv2.line(cv2_image, face_landmarks[part][0], face_landmarks[part][-1], color=(0, 0, 0), thickness=4)
            part = 'right_eye'
            for point_index in range(len(face_landmarks[part])-1):
                    cv2.line(cv2_image, face_landmarks[part][point_index], face_landmarks[part][point_index+1], color=(0, 0, 0), thickness=4)
            cv2.line(cv2_image, face_landmarks[part][0], face_landmarks[part][-1], color=(0, 0, 0), thickness=4)
        #overlaying image
        transparency_index = 0.4
        cv2_image = cv2.addWeighted(overlay, transparency_index, cv2_image, 1-transparency_index, 0)
        cv2.imwrite(args.path_output, cv2_image)



if __name__ == "__main__":
    face_makeup_main()