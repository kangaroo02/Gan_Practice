#!/usr/bin/env python
import zipfile
import cv2
import numpy as np
from numpy import savez_compressed
from PIL import Image # $ pip install pillow
from numpy import asarray

# archive = zipfile.ZipFile('train2014.zip', 'r')
# dirlist = archive.namelist()[1:]
# print(len(dirlist))
'''
print(dirlist)
print(len(dirlist))

with archive as zfile:
	data = zfile.read(dirlist[0])
imgfile = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

print(type(imgfile))
cv2.imshow("0", imgfile)
cv2.waitKey(0)
'''

# for index, filename in enumerate(dirlist):
# 	print(filename)
# 	if index == 10:
# 		break

'''
for index, filename in enumerate(dirlist):
	archive = zipfile.ZipFile('train2014.zip', 'r')
	dirlist = archive.namelist()[1:]
	with archive as zfile:
		data = zfile.read(dirlist[index])
	imgfile = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

	print(type(imgfile))
	cv2.imshow("0", imgfile)
	cv2.waitKey(0)
	if index == 10:
		break
'''

# load all images in a directory into memory
def load_images(number_limit, archive_name, size=(256,256)):
	# print(dirlist)
	src_list = list()
	archive = zipfile.ZipFile(archive_name, 'r')
	dirlist = archive.namelist()[1:]
	with archive as zfile:

		# enumerate filenames in directory, assume all are images
		for index in range(0, number_limit):
			# load and resize the image

			data = zfile.read(dirlist[index])
			imgfile = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

			resize = cv2.resize(imgfile, size)

			cv2.imshow("0", resize)
			cv2.waitKey(10)
			# pixels = img_to_array(resize)
			print(resize.shape)


			src_list.append(resize)

	return asarray(src_list)


src_images = load_images(10, 'train2014.zip')
print(src_images.shape)


filename = 'train2014_256.npz'
savez_compressed(filename, src_images)
print('Saved dataset: ', filename)