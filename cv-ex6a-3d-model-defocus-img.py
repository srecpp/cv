import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = cv2.imread("C://Users//Akash//Downloads//defocused img.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(image)
plt.title("Defocused Image")
plt.show()

laplacian = cv2.Laplacian(image, cv2.CV_64F)

laplacian_normalized = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)

depth_map = 1 - laplacian_normalized

h, w = depth_map.shape
focal_length = 1.0

x, y = np.meshgrid(np.arange(w), np.arange(h))

x = x.flatten()
y = y.flatten()
z = depth_map.flatten() * focal_length 

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='inferno', marker='.', s=0.5)  # Adjust 's' for point size

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth (Z)')
ax.set_title('3D Visualization of Depth from Defocus')

plt.show()
