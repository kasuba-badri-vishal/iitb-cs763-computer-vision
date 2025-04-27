import argparse
import cv2
import os
import pickle


def blur_image(image, x1, y1, x2, y2, ksize=(49,49), sigma=50):
    try:
        crop_image = image[y1:y2, x1:x2]
        crop_image = cv2.GaussianBlur(crop_image, ksize, sigma)
        image[y1:y2, x1:x2] = crop_image
    except:
        print(x1,y1,x2,y2)
    return image

def draw_image(image, ann_path, file, blur=False, frame=0):
    file_name = os.path.join(ann_path,os.path.splitext(file)[0] + '.txt')
    if(os.path.exists(file_name)):
        ann = pickle.load(open(file_name, 'rb'))    
        for face in ann[frame][1]:
            x1,y1,x2,y2 = face
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if(blur):
                blur_image(image, x1, y1, x2, y2)
    else:
        print("Annotations are not saved for ", file)
    return image


def draw_rectangle(event,x,y,flags,param):
    global x1,y1,x2,y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1,y1 = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        x2,y2 = x,y
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        ann.append([x1,y1,x2,y2])

def create_window(path, image_file, frame=False):
    cv2.destroyAllWindows()
    if frame.any():
        image = frame
    else:
        image = cv2.imread(os.path.join(path, image_file))
    cv2.namedWindow(image_file)
    cv2.setMouseCallback(image_file,draw_rectangle)
    return image

def save_annotations(ann_path, file, bboxes):
    result = [[], bboxes]
    output_path = os.path.join(ann_path, os.path.splitext(file)[0]) + '.txt'
    if(bboxes):
        print(bboxes)
        with open(output_path, 'wb') as output:
            pickle.dump([result], output, protocol=pickle.HIGHEST_PROTOCOL)
    return []


def annotation(input_path, ann_path):
    id = 0
    input_file_list = os.listdir(input_path)
    length = len(input_file_list)

    global ann, image
    ann = []
    image = create_window(input_path, input_file_list[id])
    
    while(True):
        cv2.imshow(input_file_list[id],image)
        k = cv2.waitKey(1) & 0xFF
        if(k==113 or k==110 or k==112):
            ann = save_annotations(ann_path, input_file_list[id], ann)
        if k == 113:
            break
        elif(k == 110):    
            id = (id+1)%length
            image = create_window(input_path, input_file_list[id])
        elif(k == 112):
            id = (id-1)%length
            image = create_window(input_path, input_file_list[id])

def drawing_boxes(input_path, ann_path, blur=False):
    input_dir = os.listdir(input_path)
    id = 0
    length = len(input_dir)
    image = create_window(input_path, input_dir[id])
    image = draw_image(image, ann_path, input_dir[id], blur=blur)

    while(True):
        cv2.imshow(input_dir[id],image)
        k = cv2.waitKey(1) & 0xFF
        if k == 113:
            break
        elif(k == 110):
            id = (id+1)%length
            image = create_window(input_path, input_dir[id])
            image = draw_image(image, ann_path, input_dir[id], blur=blur)
        elif(k == 112):
            id = (id-1)%length
            image = create_window(input_path, input_dir[id])
            image = draw_image(image, ann_path, input_dir[id], blur=blur)



def video_annotation(input_path, ann_path):
    
    vidcap = cv2.VideoCapture(input_path)
    global image, ann
    success,image = vidcap.read()
    result = []
    frames = [image]
    while success:
        frames.append(image)
        success, image = vidcap.read()

    id = 0
    image = frames[0]
    ann = []
    image = create_window(None, str(id), frame=frames[0])
    result = []
    length = len(frames)
    while(True):
        cv2.imshow(str(id),image)
        k = cv2.waitKey(1) & 0xFF
        if(k==113 or k==110 or k==112):
            result.append([[],ann])
            ann = []
        if k == 113:
            break
        elif(k == 110):    
            id = (id+1)%length
            image = create_window(None, str(id), frame=frames[id])
        elif(k == 112):
            id = (id-1)%length
            image = create_window(None, str(id), frame=frames[id])
    
    print(result)

    with open(ann_path, 'wb') as output:
        pickle.dump(result, output, protocol=pickle.HIGHEST_PROTOCOL)    

def save_blur_video(input_path, ann_path):

    vidcap = cv2.VideoCapture(input_path)
    

    data = pickle.load(open(ann_path, 'rb'))
    print(data)

    success,image = vidcap.read()
    height, width, _ = image.shape
    
    out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    id = 0


    while success and id<len(data):
        print(data[id])
        face = data[id][1]
        if(len(face)>0):
            for x1,y1,x2,y2 in face:
                image = blur_image(image, x1, y1, x2, y2)
        out.write(image)
        success,image = vidcap.read()
        id+=1

    out.release()   




def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection using Contours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data", type=str, dest='input_path', default=None, help="path to the input data")
    parser.add_argument("--annotation", type=str, dest='ann_path', default=None, help="path to the annotation data")
    parser.add_argument("--type", type=int, default=None, dest='input_type', help="type of the input data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if(args.input_type==1):
        annotation(args.input_path, args.ann_path)
    elif(args.input_type==2):
        drawing_boxes(args.input_path, args.ann_path, blur=False)
    elif(args.input_type==3):
        drawing_boxes(args.input_path, args.ann_path, blur=True)
    elif(args.input_type==4):
        video_annotation(args.input_path, args.ann_path)
    elif(args.input_type==5):
        save_blur_video(args.input_path, args.ann_path)
    