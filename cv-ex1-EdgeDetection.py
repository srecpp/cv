import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image, title=''):
    if image is None:
        print(f"Error: Unable to load image")
        return
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# Replace with your actual image path
image_path = "ex1-2input.jpg"
image = cv2.imread(image_path)

if image is not None:
    display_image(image, "Original Image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Save the edges detected image
    cv2.imwrite("edges_detected.jpg", edges)

    # Display edges
    display_image(edges, "Edges Detected")
else:
    print(f"Error: Unable to load image at path: {image_path}")
