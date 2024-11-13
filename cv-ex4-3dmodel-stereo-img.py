import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img_left = cv2.imread("C://Users//Akash//Downloads//left.png", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread("C://Users//Akash//Downloads//right.png", cv2.IMREAD_GRAYSCALE)

min_disparity = 0
num_disparities = 64  # Must be divisible by 16
block_size = 9        # Must be an odd number

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,   # Control for smooth disparity map
    P2=32 * 3 * block_size ** 2,  # Control for smooth disparity map
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=50,
    speckleRange=2
)

disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.imshow(disparity_normalized, cmap='plasma')
plt.colorbar()
plt.title("Disparity Map")
plt.show()

focal_length = 0.8 
baseline = 0.1 

depth_map = (focal_length * baseline) / (disparity + 1e-6)

h, w = depth_map.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))

x = x.flatten()
y = y.flatten()
z = depth_map.flatten()
color = disparity_normalized.flatten()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=color, cmap='plasma', marker='.', s=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth (Z)')
ax.set_title('3D Point Cloud from Stereo Images')

plt.show()
