import argparse
import cv2
import face_recognition
import numpy as np
import os


def create_window(image_file, frame=[]):
    cv2.destroyAllWindows()
    global image
    image = cv2.imread(image_file)
    cv2.namedWindow(image_file)
    return image


def initialize_data(args):
    known_face_encodings = []

    if(args.type==2 or ((args.type==3) and ('captured' in args.inp_file))):
        known_face_names = ['person1', 'person2']
        img_dir = './../data/captured/'
    else:
        known_face_names = ['arnold', 'sylvester']
        img_dir = './../data/samples/'

    for name in known_face_names:
        person_image = face_recognition.load_image_file(img_dir+name+".jpg")
        face_encoding = face_recognition.face_encodings(person_image)[0]
        known_face_encodings.append(face_encoding)

    return known_face_names, known_face_encodings

def recognize_people(args):

    known_face_names, known_face_encodings = initialize_data(args)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    if(args.type==3):
        video_capture = cv2.VideoCapture(args.inp_file)
        ret, image = video_capture.read()
        height, width, _ = image.shape
        output = cv2.VideoWriter(args.out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    else:
        image = create_window(args.inp_file)

    # print(image.shape)
    while(True):

        # Only process every other frame of video to save time
        if process_this_frame:

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)               
                if(matches[best_match_index]):
                    name = known_face_names[best_match_index]
                if(abs(face_distances[0]-face_distances[1])<0.09):
                    name = "Unknown"
                face_names.append(name)

        process_this_frame = not process_this_frame

        
        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left, bottom+36 ), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if(image.shape[0]>=1000):
                cv2.putText(image, name, (left, bottom+24), font, 1.5, (255, 255, 255), 1)
            else:
                cv2.putText(image, name, (left, bottom+16), font, 0.8, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Image', image)


        if(args.type == 3):
            output.write(image)
            ret, image = video_capture.read()
            if(ret==False):
                break
        else:
            while(True):
                cv2.imshow('Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if(args.type==3):
        output.release()
        video_capture.release()
    else:
        cv2.imwrite(args.out_file, image)

    cv2.destroyAllWindows()



def parse_args():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser(description="Face Recognition", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("-i", type=str, dest='inp_file', default=None, help="path to the input data file")
    parser.add_argument("-o", type=str, dest='out_file', default=None, help="path to the output data file")
    parser.add_argument("--type", type=int, dest='type', default=1, help="type of the input data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ### Create Results directory if it is not existing
    directory = os.path.split(args.out_file)
    if( not os.path.exists(directory[0])):
         os.makedirs(directory[0])

    ### Run the main program
    recognize_people(args)
