import cv2 as cv
import numpy as np
import os
import argparse
import xml.etree.ElementTree as ET



def main():

	parser = argparse.ArgumentParser(prog="resize images",
		description="Resize images to the specified size",
		epilog="Thank you for using it!",
		fromfile_prefix_chars="@")

	parser.add_argument("-f", "--file", required=True, help="Path to xml file")
	parser.add_argument("-d", "--dir", required=True, help="Directory of Images to be reshaped")
	parser.add_argument("-r", "--results", required=True, help="Directory for saving the resultant Images")

	parser.add_argument("-s", "--size", default=None, help="Size of the new Image")

	args = parser.parse_args()


	tree = ET.parse(args.file)

	root = tree.getroot()

	if args.size == None:
		size = input("Enter the size of new Image with aspect ratio of 1.0:" )
		if args.size == None:
			size = int(size)
		else:
			size = int(args.size)


	if not os.path.exists(args.dir):
		raise Systemexit(f"Directory of Images don't exists: {args.dir}")


	if not os.path.exists(args.results):
		try:
			os.makedirs(args.results)
		except Oserror:
			raise Systemexit(f"Reuslts Directory don't exists: {args.results}")


	files = os.listdir(args.dir)


	for file in files:

		image = root.find(".//image[@file='{}']".format(file))
		# print(image)
		file_name = image.attrib['file']
		new_filename = f"{file_name}"
		image.set('file', new_filename)
		image.set('width', '224')
		image.set('height', '224')
		# print('File:', file_name)
		# print('Box coordinates:', top, left, width, height)


		file_loc = os.path.join(args.dir, file)

		img = cv.imread(file_loc)

		h, w, _ = img.shape

		ratio = w/h

		if h>w:
			new_h = (size)
			new_w = int(new_h*ratio)
		else:
			new_w = (size)
			new_h = int(new_w/ratio)

		resized_img = cv.resize(img, (new_w, new_h))

		scaling_len = new_h/h
		scaling_width = new_w/w
		delta_w = size - new_w
		delta_h = size - new_h
		
		top, bottom = delta_h//2, new_h + (delta_h//2)
		left, right = delta_w//2, new_w + (delta_w//2)
		color = [0,0,0]

		# box = image.find('box')
		for box in image.findall("box"):

			box_top 	= int(box.attrib['top'])
			box_left 	= int(box.attrib['left'])
			box_width 	= int(box.attrib['width'])
			box_height	= int(box.attrib['height'])
			

			box_left 	= int(box_left*scaling_width) + (delta_w//2)
			box_width 	= int(box_width*scaling_width)
			box_top 	= int(box_top*scaling_len) + (delta_h//2)
			box_height	= int(box_height*scaling_len)

			box.set('left', str(box_left))
			box.set('top', str(box_top))
			box.set('width', str(box_width))
			box.set('height', str(box_height))

		padded_img = np.zeros((size,size,3))

		padded_img[top:bottom, left:right, :] = resized_img

		output_path = os.path.join(args.results, file)

		cv.imwrite(output_path, padded_img)
		print(f"Writen: {file}")

	tree.write('training_renamed.xml')


if __name__ == "__main__":
	main()