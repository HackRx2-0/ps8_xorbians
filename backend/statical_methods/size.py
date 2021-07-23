import cv2
import numpy as np

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