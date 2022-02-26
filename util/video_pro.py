import cv2
import os
import glob
import numpy as np

path = os.getcwd()
os.chdir(path)

frame_rate = 29
frame_height = 240
frame_width = 352
num_frames = 999

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('data/subtraction_out.avi', fourcc, frame_rate, (frame_width*2, frame_height))


fname1 = glob.glob("output003/*.jpg")
fname2 = glob.glob("output/*.jpg")


for i in range(num_frames):
    img1 = cv2.imread(fname1[i])
    img2 = cv2.imread(fname2[i])
    img = np.concatenate((img1, img2), axis=1)
    video.write(img)
    cv2.imwrite('output004/out'+str(i)+'.png', img)

cv2.destroyAllWindows()
video.release()
