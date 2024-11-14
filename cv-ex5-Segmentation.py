import cv2 
import numpy as np
from matplotlib import pyplot as plt
image_path = 'multi.jpg' 
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
threshold_value = 128 
ret, binary_mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold_value, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb) 
plt.title('Original Image') 
plt.axis('off')
plt.subplot(1, 2, 2) 
plt.imshow(binary_mask, cmap='gray') 
plt.title('Binary Mask')
plt.axis('off')
plt.tight_layout() 
plt.show()
