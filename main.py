import cv2
import numpy as np

# Load the image
img = cv2.imread("people.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment out the people
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize the list of detected people
people = []

# Iterate through the contours
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if area > 1000 and aspect_ratio > 0.5 and aspect_ratio < 2:
        people.append((x, y, w, h))

# Print the number of people detected
print("Number of people detected:", len(people))

# Draw bounding boxes around the detected people
for person in people:
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
