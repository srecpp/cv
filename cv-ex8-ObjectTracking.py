import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load video and cascade classifier
capture = cv2.VideoCapture('ex8input.mp4')
car_cascade = cv2.CascadeClassifier('cars.xml')

def show_frame(frame, title="Frame"):
    """Display the frame using Matplotlib."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

frame_count = 0
while True:
    ret, frames = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(frames, (x, y - 40), (x + w, y), (51, 51, 255), -2)
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Resize the frame for display
    frames = cv2.resize(frames, (600, 400))

    # Show the frame with detected cars
    show_frame(frames, "Detected Cars")

    # Increment frame counter
    frame_count += 1

    # Break the loop after 50 frames
    if frame_count > 50:
        break

# Release the video capture object
capture.release()
