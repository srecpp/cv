import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the video file
cap = cv2.VideoCapture("exyinput.mp4")

# Read the first frame of the video
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read video file")

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize a mask to store optical flow visualization
mask = np.zeros_like(first_frame)
mask[..., 1] = 255  # Set the saturation to 255 (max value)

# Create figure for displaying the frames and optical flow
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# List to store the frames and optical flow visualizations
frames = []
optical_flows = []

# Process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert the flow into magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update the mask with the angle and normalized magnitude for visualization
    mask[..., 0] = angle * 180 / np.pi / 2  # Hue channel (angle)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value channel (magnitude)

    # Convert the mask to BGR color format for visualization
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Store the original frame and the optical flow visualization
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    optical_flows.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    # Set the previous frame for the next iteration
    prev_gray = gray

# Release the video capture object
cap.release()

# Display optical flow for a few selected frames
for i in range(10, 15):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(frames[i])
    ax[0].set_title("Original Frame")
    ax[0].axis("off")
    
    ax[1].imshow(optical_flows[i])
    ax[1].set_title("Optical Flow Frame")
    ax[1].axis("off")
    
    plt.show()
