import cv2
import numpy as np

def check_blur(img):
	sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
	if np.std(sobelxy)<50:
		blur = True
	else:
		blur = False
	return sobelxy, blur, 1/(1+np.exp(50-np.std(sobelxy)))