from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cv2.cv as cv
import cv2
from PIL import Image
import math

import re
#Methods to sort as human expect
def tryint(s):
    try:
        return int(s)
    except:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
#-----------------------------------------------------------------

#CUT THE IMAGES
def cutRetina(img):
    "Cuttig - Receive a RGB image in cv2 format"
    #img = image_list[60]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #gray = img.convert('L') 

    #gray = np.asarray(gray)

    #plt.imshow(gray,cmap='gray')
    #plt.show()

    r, c  = gray.shape

    thresh = 5

    r1 = 0; c1 = 0; r2 = r; c2 = c
    
    #print ("r1---------------")
    for i in xrange(r/2, 0, -1):
        if np.mean(gray[i,:]) <= thresh:
            r1 = i
            break
    #print ("c1----------------------")
    for i in xrange(c/2, 0, -1):
        if np.mean(gray[:,i]) <= thresh:
            c1 = i
            break
    #print ("r2----------------------")
    for i in xrange(r/2, r, 1):
        if np.mean(gray[i,:]) <= thresh:
            r2 = i
            break

    #print ("c2----------------------")
    for i in xrange(c/2, c, 1):
        if np.mean(gray[:,i]) <= thresh:
            c2 = i
            break

    corpIm = img[r1:r2,c1:c2]  #corpIm= img.crop((c1, r1, c2 , r2 )) #x,y,x2,y2

    #BGR to RGB --> OpenCV image to Matplotlib ---> Commented for the new gray pre procesing
    #corpIm = corpIm[:,:,::-1]

    #plt.imshow(corpIm)
    #plt.show()
    return corpIm



#MAKE THE IMAGE A SQUARE ADDING ZEROS AT THE MATRIX
def squareImage(img1):

	#Initial conditions
	top = bottom = left = rigth = 0

	BLACK = [0,0,0]
	h, w , _ = img1.shape
	dif = abs(h-w)

	#Verify in witch senses add the border
	if dif != 0:
	    if h > w:
	        #Needs to add in left-rigth
	        left = dif/2
	        rigth  = dif/2 + (dif % 2) 
	    else:
	        #Needs to add in top-bottom
	        top = dif/2
	        bottom  = dif/2 + (dif % 2)  

	#Make border
	constant= cv2.copyMakeBorder(img1,top,bottom,left,rigth,cv2.BORDER_CONSTANT,value=BLACK)

	return constant

#DATA CENTRALIZATION AND NORMALIZATION 
def centerAndNormX(X_train , mean_RGB , std_RGB):
   
    X_train = X_train.astype('float32')
    X_train[:,0,:,:] = X_train[:,0,:,:] - mean_RGB[0]
    X_train[:,1,:,:] = X_train[:,1,:,:] - mean_RGB[1]
    X_train[:,2,:,:] = X_train[:,2,:,:] - mean_RGB[2]

    X_train[:,0,:,:] = X_train[:,0,:,:]/std_RGB[0]
    X_train[:,1,:,:] = X_train[:,1,:,:]/std_RGB[1]
    X_train[:,2,:,:] = X_train[:,2,:,:]/std_RGB[2]
    
    return X_train

#HISTOGRAM EQUALIZATION
def histogramaEq(img):
   
	#Equialize each chanel
	img[:, :, 0] = cv2.equalizeHist(img[:, :, 0]) 
	img[:, :, 1] = cv2.equalizeHist(img[:, :, 1]) 
	img[:, :, 2] = cv2.equalizeHist(img[:, :, 2]) 

	return img

#CENTRED THE DATA AND NORMALIZATION
def centerAndNormData(I):
	#Mean subtraction -> cero centered data
	Y = I- np.mean(I)

	#Normalization
	X = Y /np.std(Y)

	return X
