import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_background(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    background = get_background(file_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    consecutive_frame = 10
    frame_diff_list = []
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []
        
        frame_diff = cv2.absdiff(gray, background)
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        frame_diff_list.append(dilate_frame)
        
        if len(frame_diff_list) == consecutive_frame:
            sum_frames = sum(frame_diff_list)
            contours, _ = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            processed_frames.append(orig_frame)

        if len(processed_frames) == 20:  # Capture only the first n frames
            break

    cap.release()
    return processed_frames

# Process the video and get the first 5 frames
file_path = 'input.mp4'
frames = process_video(file_path)

# Visualize the frames in Jupyter Notebook
plt.figure(figsize=(5, 40))
for i, frame in enumerate(frames):
    plt.subplot(20, 1, i + 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.axis('off')  # Turn off axis

plt.show()
