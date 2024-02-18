import cv2
import csv
import os
import datetime
import face_recognition

# Load known faces and their names from the 'faces' folder
known_faces = []
known_names = []

for filename in os.listdir('faces'):
    image = face_recognition.load_image_file(os.path.join('faces', filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

video_capture = cv2.VideoCapture(0)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]

        # Use face recognition library to compare the detected face with known faces
        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]

        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        with open('attendance.csv', 'r') as file:
            reader = csv.reader(file)
            existing_names = set()
            for row in reader:
                if row:  # Check if the row is not empty
                    existing_names.add(row[0])

        with open('attendance.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if name not in existing_names:
                writer.writerow([name, current_time])
                existing_names.add(name)

        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

