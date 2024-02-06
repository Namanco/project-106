import cv2

# Load the pre-trained Haar Cascade classifier for detecting people
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Open a video file (replace with your video path)
video_path = "PRO-106-ProjectTemplate-main/walking.avi"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()


    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the frame
    people = person_cascade.detectMultiScale(gray,1.1,5)

    # Draw bounding boxes around detected people
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Display the frame with detected people
    cv2.imshow("People Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(25) == 32:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()