import pickle
import os
import cv2 as cv
import numpy as np
from gaussian_v2 import GMM, Gaussian, Pixel

wd = os.getcwd()
os.chdir(wd)


with open('gmm.pickle', 'rb') as f:
    gmm = pickle.load(f)

index = 1
fname = ["https://github.com/sngirish/background_subtraction/blob/main/data/umcp.mpg?raw=true"]
cap = cv.VideoCapture(fname[0])
while cap.isOpened():
    print("Frame # being read : ", int(cap.get(cv.CAP_PROP_POS_FRAMES)))        
    success, img = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):      
            l = list(img[i][j])
            pixel = Pixel(l[0], l[1], l[2])                
            pixel = gmm[i][j].predict(pixel)
            img[i][j] = np.array(list(pixel))
    cv.imwrite(r'output001/'+'%05d'%index+'.jpg', img)
    index += 1
cap.release()


    
