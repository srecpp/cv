import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

# Load the image
image = cv2.imread("3d-i.jpg")

# Display the original image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect edges using the Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# Use edges as a simple depth map
depth_map = edges

# Get the coordinates and depth values of non-zero points
rows, cols = np.where(depth_map > 0)
depth_values = depth_map[depth_map > 0]
points = np.column_stack((cols, rows, depth_values))

# Plot the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
cbar = plt.colorbar(scatter)
cbar.set_label('Depth value')

plt.show()