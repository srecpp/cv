import cv2
import sys
import matplotlib.pyplot as plt

imagePath = "cv9input.jpg"
cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

image_resized = cv2.resize(image, (800, 600))

image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Faces Detected")
plt.axis("off")
plt.show()
