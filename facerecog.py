import cv2
import face_recognition
import os
import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Load the required trained XML classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load and encode known faces
known_face_encodings = []
known_face_names = []

# Load images and create encodings (put your images in a folder named "known_faces")
known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png")):
        img_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use the file name (without extension) as the person's name
        else:
            print(f"No faces found in {filename}")

# Capture frames from a camera using DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variable to keep track of the last spoken name to avoid repeating speech
last_spoken_name = ""

# Loop runs if capturing has been initialized
while True:
    # Reads frames from the camera
    ret, img = cap.read()

    if not ret:
        print("Failed to capture image from the camera")
        break

    # Convert to gray scale of each frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascades
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect faces using face_recognition
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (x, y, w, h), (top, right, bottom, left), face_encoding in zip(faces, face_locations, face_encodings):
        # To draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Check if the face is a match for known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if face_distances.size > 0:  # Ensure face_distances is not empty
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # If the recognized name is different from the last spoken name, speak the welcome message
                if name != last_spoken_name:
                    speech_text = f"Welcome {name}"
                    engine.say(speech_text)
                    engine.runAndWait()
                    last_spoken_name = name

        # Draw the name below the face
        cv2.putText(img, name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # To draw a rectangle around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 127, 255), 2)

    # Display the resulting image
    cv2.imshow('img', img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
