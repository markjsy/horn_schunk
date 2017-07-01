import cv2
import random
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import normalize
from matplotlib.pyplot import figure,subplots,draw,pause,gca,show

np.set_printoptions(linewidth = 1000)
np.set_printoptions(threshold=np.nan)

def compute_dx(img):
	mask = np.array([[-1,1],
					 [-1,1]])
	dx = ndimage.convolve(img,mask)
	return dx

def compute_dy(img):
	mask = np.array([[-1,-1],
					 [ 1, 1]])
	dy = ndimage.convolve(img,mask)
	return dy

def compute_fx(img_1, img_2):
	mask = np.array([[-1,1],
					 [-1,1]])
	dx_1 = ndimage.convolve(img_1, mask)
	dx_2 = ndimage.convolve(img_2, mask)
	dx = dx_1 + dx_2
	return dx

def compute_fy(img_1, img_2):
	mask = np.array([[-1,-1],
					 [ 1, 1]])
	dy_1 = ndimage.convolve(img_1, mask)
	dy_2 = ndimage.convolve(img_2, mask)
	dy = dy_1 + dy_2
	return dy

def compute_ft(img_1, img_2):
	mask_1 = np.array([[ -1,  -1],
					   [ -1,  -1]])

	mask_2 = np.array([[ 1,  1],
					   [ 1,  1]])

	dt_1 = ndimage.convolve(img_1, mask_1)
	dt_2 = ndimage.convolve(img_2, mask_2)
	dt = dt_1 + dt_2
	return dt

def laplacian(fx, fy):
	mask = np.array([[  0, -1,  0],
					 [ -1,  4, -1],
					 [  0, -1,  0]]) * (1.0/4.0)
	fxx = compute_dx(fx)
	fyy = compute_dy(fy)

	fxy = fxx + fyy

	lap = ndimage.convolve(fxy, mask)

	return lap

def horn_schunk(img_1, img_2, k, L):
	u = 0
	v = 0

	fx = compute_fx(img_1, img_2)
	fy = compute_fy(img_1, img_2)
	ft = compute_ft(img_1, img_2)

	P = (fx*u) + (fy*v) + ft
	D = laplacian(fx, fy)

	u_av = u - fx * (P/D)
	v_av = v - fy * (P/D)

	for i in range(1, k):
		fx = compute_fx(u_av, v_av)
		fy = compute_fy(u_av, v_av)
		ft = compute_ft(u_av, v_av)
		P = (fx*u_av) + (fy*v_av) + ft
		D = laplacian(fx, fy)

		u_av = u_av - fx * (P/D)
		v_av = v_av - fy * (P/D)
	
	return u_av/L,v_av/L

def dispOpticalFlow(Image, u, v, Divisor, name ):
	'''
	Display image with a visualisation of a flow over the top. 
	A divisor controls the density of the quiver plot."
	'''

	PictureShape = np.shape(Image)
	#determine number of quiver points there will be
	Imax = int(PictureShape[0]/Divisor)
	Jmax = int(PictureShape[1]/Divisor)
	#create a blank mask, on which lines will be drawn.
	mask = np.zeros_like(Image)
	for i in range(1, Imax):
	  for j in range(1, Jmax):
	     X1 = (i)*Divisor
	     Y1 = (j)*Divisor
	     X2 = int(X1 + u[X1,Y1])
	     Y2 = int(Y1 + v[X1,Y1])
	     X2 = np.clip(X2, 0, PictureShape[0])
	     Y2 = np.clip(Y2, 0, PictureShape[1])


	     if(X1 != X2):
	     	print str(X1) + " " + str(X2)
	     	print str(Y1) + " " + str(Y2)

	     #add all the lines to the mask
	     mask = cv2.line(mask, (Y1,X1),(Y2,X2), [255, 255, 255], 1)
	#superpose lines onto image
	img = cv2.add(Image,mask)
	#print image
	cv2.imshow(name,img)
	return []

