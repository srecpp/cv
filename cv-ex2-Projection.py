import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None

    # Define the source and destination points for the perspective transform
    pts1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])

    # Get the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective warp
    result = cv2.warpPerspective(image, matrix, (500, 600))

    return image, result

def display_image(image, title=''):
    if image is None:
        print(f"Error: Unable to load image.")
        return
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Path to the image file
image_path = "ex1-2input.jpg"

# Perform the perspective transform
original, transformed = perspective_transform(image_path)

if original is not None and transformed is not None:
    # Display the original and transformed images
    display_image(original, 'Original Image')
    display_image(transformed, 'Transformed Image')
