import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("ex1-2input.jpg")

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set up the plotting environment
plt.figure(figsize=(10, 10))

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Apply a small blur (5x5)
image_blurred = cv2.blur(src=image, ksize=(5, 5))
image_blurred_rgb = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB)

# Plot the blurred image
plt.subplot(1, 3, 2)
plt.imshow(image_blurred_rgb)
plt.title('Blurred Image (5x5)')
plt.axis('off')

# Apply a large blur (75x75)
image_blurred_large = cv2.blur(src=image, ksize=(75, 75))
image_blurred_large_rgb = cv2.cvtColor(image_blurred_large, cv2.COLOR_BGR2RGB)

# Plot the large blurred image
plt.subplot(1, 3, 3)
plt.imshow(image_blurred_large_rgb)
plt.title('Blurred Image (75x75)')
plt.axis('off')

# Show all plots
plt.show()