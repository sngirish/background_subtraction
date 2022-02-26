import os
import glob
import cv2 as cv
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial import distance
from scipy.stats import multivariate_normal as mv_norm
import collections
import pickle


class Pixel():
    def __init__(self, R=0, G=0, B=0):
        self.R = R
        self.G = G
        self.B = B
        self.data = [self.R, self.G, self.B]

    def __iter__(self):
        return iter(self.data)

    def set_rgb(self, R, G, B):
        setattr(self, 'R', R)
        setattr(self, 'G', G)
        setattr(self, 'B', B)
        self.data = [self.R, self.G, self.B]

class Gaussian():
    def __init__(self, mu = np.zeros(3), sigma = 225*np.eye(3)):
        self.mu = mu
        self.sigma = sigma
    
    def check(self, pixel):
        x = np.array(list(pixel), dtype=float)
        u = self.mu
        
        d = distance.mahalanobis(x, u, inv(self.sigma))
        """
        delta = np.array(list(pixel), dtype=float) - self.mu
        d = sum(delta*delta/np.diag(self.sigma))
        """
        if d < 2.5:            
            return True
        else:
            return False

class GMM():
    def __init__(self, mean_vec, K=4, w = [0.7, 0.11, 0.1, 0.09], T = 0.9):
        self.K = K
        self.weights = np.array(w)
        self.M = [Gaussian(mu = mean_vec) for i in range(self.K)]
        self.B = None
        self.T = T
        
    def update_weights(self, pixel, alpha):
        match = -1
        for i in range(self.K):
            if self.M[i].check(pixel):
                match = i
                break
        
        if match != -1:
            x = np.array(list(pixel), dtype=float)
            u = self.M[i].mu
            delta = x - u           
            rho = alpha * mv_norm.pdf(x, u, self.M[i].sigma)
            
            self.weights[i] = (1 - alpha) * self.weights[i]
            self.weights[i] += alpha         
            self.M[i].mu += rho * delta
            sig = self.M[i].sigma
            self.M[i].sigma = sig + rho * (np.matmul(delta, delta.T) - sig)
            self.M[i].sigma = self.M[i].sigma*np.eye(3)

        if match == -1:
            id = np.argmin(self.weights)
            self.M[id].mu = np.array(list(pixel), dtype=float)      #Mean equal to current value
            self.M[id].sigma = 225*np.eye(3) #High variance

    def norm1(self):
        return np.array([np.sqrt(norm(self.M[k].sigma)) for k in range(self.K)])
    
    def reorder(self):
        ratio = self.weights/self.norm1()
        order_idx = np.argsort(-ratio)

        self.weights = self.weights[order_idx]
        self.M = list(np.array(self.M)[order_idx])

        cum_wt = 0
        for index in order_idx:
            cum_wt += self.weights[index]
            if cum_wt > self.T:
                self.B = index + 1
                break

    def predict(self, pixel):      
        for k in range(self.B):
            if self.M[k].check(pixel):
                pixel.set_rgb(255, 255, 255)                
                break
        return pixel

if __name__=='__main__':
    fname = ["https://github.com/sngirish/background_subtraction/blob/main/data/umcp.mpg?raw=true"]
    cap = cv.VideoCapture(fname[0])
    if cap.isOpened():
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        
        fps = int(cap.get(cv.CAP_PROP_FPS))
        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f'Video frame height: {h:d}')
    print(f'Video frame width: {w:d}')
    print(f'Video frame frame rate: {fps:d}')
    print(f'Video frame total number of frames: {n_frames:d}')
    
    alpha = 0.01
    
    cap = cv.VideoCapture(fname[0]) 
    while cap.isOpened():
        index = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        print("Frame # being read : ", index)
        
        success, img = cap.read()
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if index == 0:
            gmm = [[GMM(np.array(img[j][i], dtype=float)) for i in range(img.shape[1])]
           for j in range(img.shape[0])]
            
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):      
                l = list(img[i][j])
                pixel = Pixel(l[0], l[1], l[2])
                gmm[i][j].update_weights(pixel, alpha)
                gmm[i][j].reorder()

    cap.release()        

    index = 0
    cap = cv.VideoCapture(fname[0])
    while cap.isOpened():
        print("Frame # being read : ", int(cap.get(cv.CAP_PROP_POS_FRAMES))-1)        
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
        #cv.imwrite(r'./output003/'+'%05d'%index+'.jpg', img)
        index += 1
    cap.release()

    #pickle.dump(gmm, file = open("gmm.pickle", "wb"))
    

                                           

        
