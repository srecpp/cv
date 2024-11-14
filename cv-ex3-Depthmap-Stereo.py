import numpy as np 
import cv2
from matplotlib import pyplot as plt
imgL = cv2.imread('left01.jpg', 0) 
imgR = cv2.imread('right01.jpg', 0)
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
disparity = stereo.compute(imgL, imgR)
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(disparity_normalized, 'gray') 
plt.show()
