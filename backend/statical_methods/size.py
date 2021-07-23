import cv2
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import average

def check_dimensions(orig, img, base_size=0.2, lim=10000):
	area_orig = orig.shape[0]*orig.shape[1]
	area_face = img.shape[0]*img.shape[1]
	scale_percent = 1.0
	if (area_face/area_orig) < base_size and area_face<lim:
		small = True
		scale_percent = (area_orig/area_face)**(0.5)
		width = int(img.shape[1] * scale_percent)
		height = int(img.shape[0] * scale_percent)
		dim = (width, height)
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	else:
		small = False
		resized = img
	return resized, small, scale_percent

def aspect_ratio(orig, bbox,scale_percent, ratio = [4,3]):
	x,y,x1,y1 = bbox
	x_cent = (x+x1)/2
	y_cent = (y+y1)/2
	h = y1-y
	w = x1-x

	if h<w:
		w = ratio[1]*h/ratio[0]
	else:
		h = ratio[0]*w/ratio[1]

	x = x_cent - w/2
	x = int(x)
	if x < 0:
		x = 0
	x1 = x_cent + w/2
	x1 = int(x1)
	if x1 > orig.shape[0]:
		x1 = orig.shape[0]

	y = y_cent - h/2
	y = int(y)
	if y < 0:
		y = 0
	y1 = y_cent + h/2
	y1 = int(y1)
	if y1 > orig.shape[0]:
		y1 = orig.shape[0]
	img = orig[y:y1,x:x1,:]
	width = int(img.shape[1] * scale_percent)
	height = int(img.shape[0] * scale_percent)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized
	#return 
	
	
