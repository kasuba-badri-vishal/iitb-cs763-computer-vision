import numpy as np
import cv2 as cv
import argparse
import os


def CheckFileExists(path):
    """
    Check if the given file exists or not
    path : Relative Path of file
    """
    # Check if the file exists at the given path, and raise an error if it doesn't
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")


def getLabel(age):

	if age <= 13:
		return "child"
	elif age <= 19:
		return "teens"
	elif age <= 28:
		return "iitians"
	elif age <= 40:
		return "middle"
	elif age <= 55:
		return "boomers"
	elif age <= 65:
		return "downhill"
	elif age <= 75:
		return "senior"
	else:
		return "super"



def main():

	parser = argparse.ArgumentParser(prog = "Apparent Age model",
		description="For parsing the model and prototxt file",
		epilog="Thank you for using!!",
		allow_abbrev=True,
		fromfile_prefix_chars="@")

	parser.add_argument("-i", "--image", required=True, help="Path to Image")

	# parser.add_argument("-m", "--model", required=True, help="Path to model")
	# parser.add_argument("-p", "--prototxt", required=True, help="Path to prototxt testing file")


	#Model location hardcoded
	model1_path 	= "Model/dex_chalearn_iccv2015.caffemodel"
	prototxt1_path 	= "Model/age.prototxt"


	model2_path 	= "Model/gender.caffemodel"
	prototxt2_path 	= "Model/gender.prototxt"


	args = parser.parse_args()

	CheckFileExists(args.image)

	#Load the image
	img = cv.imread(args.image)

	# model = cv.dnn.readNetFromCaffe(args.prototxt, args.model)

	#use the “read” methods and load a serialized age model from disk directly
	model1 = cv.dnn.readNetFromCaffe(prototxt1_path, model1_path)

	#use the “read” methods and load a serialized Gender model from disk directly
	model2 = cv.dnn.readNetFromCaffe(prototxt2_path, model2_path)

	#Detect faces out from the image with 40% margin on each side

	#Converting image from BGR to GRAY
	gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	#Getting the Haar Cascade from xml file
	haarcascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

	vj_faces_cords = haarcascade.detectMultiScale(gray_img, 1.4, 6)

	for (x,y,w,h) in vj_faces_cords:

		# #calculate face margin
		# x_margin = int(w*0.4)
		# y_margin = int(h*0.4)

		face = img[y:y+h, x:x+w]
		# face = img[y-y_margin:y+h+y_margin, x-x_margin:x+w+x_margin]

		cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

		#From the prototxt 
		"""
		input_dim: 1 - specifies the batch size of the input data.

		input_dim: 3 - specifies the number of color channels in the input image (3 for RGB images).

		input_dim: 224 - specifies the height of the input image in pixels.

		input_dim: 224 - specifies the width of the input image in pixels.
		"""

		#Creating Blob

		"""
		Normalize the data and remove illumination effect by mean subtraction based on ImageNet dataset

		The mean values for the three color channels are:

		Red: 123.68
		Green: 116.779
		Blue: 103.939

		"""	

		meanValues = (104,117,123)
		width  = 224
		height = 224

		#After executing this command our "blob" now has the shape: (1, 3, 224, 224)
		blob = cv.dnn.blobFromImage(face, 1, (height, width), meanValues)

		# set the blob as input to the network1 and perform a forward-pass to
		model1.setInput(blob)
		preds1 = model1.forward()
		
		# set the blob as input to the network2 and perform a forward-pass to
		model2.setInput(blob)
		preds2 = model2.forward()

		# calculate the expected value by taking the weighted average of the age labels
		predicted_age = np.sum(np.arange(101) * preds1)

		predicted_age = f"{predicted_age:.2f}"

		label = getLabel(float(predicted_age))

		if preds2[0][0] > 0.5:
			gender = "F"
		else:
			gender = "M"

		text = f"({gender}, {label})"

		font = cv.FONT_HERSHEY_PLAIN
		cv.putText(img, text, (x + 4, y+h + 15), font, 0.8, (0, 0, 255), 1)

		# print the predicted age
		print(f"(Gender, Predicted Age): {text}")

	filename = os.path.basename(args.image)
	result_dir = "../results/age-gender/"

	if not os.path.exists(result_dir):
		try:
			os.makedirs(result_dir)
		except Oserror:
			raise Systemexit(f"Directory not found")

	output_loc = os.path.join(result_dir, filename)
	cv.imwrite(output_loc, img)

	cv.imshow("Image", img)
	cv.waitKey()
	cv.destroyAllWindows()



if __name__ == "__main__":
	main()


