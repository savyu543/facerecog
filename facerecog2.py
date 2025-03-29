import threading
import cv2
import os
import time
from datetime import datetime, timedelta
import face_recognition
import numpy as np
import mysql.connector
import logging
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MySQL database connection setup
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'face_recognition_db'
}

detected_faces_folder = r'C:\Users\vanch\OneDrive\Desktop\detected_faces'
unknown_faces_folder = r'C:\Users\vanch\OneDrive\Desktop\unknown_faces'  # New folder for unknown faces

# Ensure the unknown_faces folder exists
if not os.path.exists(unknown_faces_folder):
    os.makedirs(unknown_faces_folder)

def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        logging.error(f"Database connection error: {err}")
        raise

# Function to load known faces and their encodings
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_labels = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(known_faces_dir, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_labels.append(os.path.splitext(filename)[0])
            except Exception as e:
                logging.error(f"Error loading face image {filename}: {e}")
    return known_face_encodings, known_face_labels

# Load known faces
known_faces_dir = 'known_faces'
known_face_encodings, known_face_labels = load_known_faces(known_faces_dir)

# Function to save unauthorized alert to the database
def save_unauthorized_alert(description):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        alert_date = datetime.now().strftime('%Y-%m-%d')
        alert_time = datetime.now().strftime('%H:%M:%S')
        
        cursor.execute('''
        INSERT INTO unauthorized_alerts (alert_date, alert_time, description)
        VALUES (%s, %s, %s)
        ''', (alert_date, alert_time, description))
        conn.commit()
        
        logging.info(f"Unauthorized alert saved: {description}")
    
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
    finally:
        if conn:
            conn.close()

# Function to record attendance for a user
def record_attendance(user_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute('''
        INSERT INTO attendance (user_id, date)
        VALUES (%s, %s)
        ''', (user_id, date))
        conn.commit()
        
        logging.info(f"Attendance recorded for user_id: {user_id}")
    
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
    finally:
        if conn:
            conn.close()

# Function to save recognized face to the database
def save_to_recognized_db(name, entry_time):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if the face has been recognized in the last 12 hours
        twelve_hours_ago = datetime.now() - timedelta(hours=12)
        cursor.execute('''
        SELECT * FROM recognized_faces
        WHERE name = %s AND entry_time >= %s
        ''', (name, twelve_hours_ago.strftime('%Y-%m-%d %H:%M:%S')))
        result = cursor.fetchone()
        
        if not result:
            cursor.execute('''
            INSERT INTO recognized_faces (name, entry_time) VALUES (%s, %s)
            ''', (name, entry_time))
            conn.commit()
            logging.info(f"Saved to recognized database: {name} at {entry_time}")
    
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
    finally:
        if conn:
            conn.close()

# Function to save unknown face to the database and move the image to the unknown_faces folder
def save_to_unknown_db(face_img, entry_time):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Save unknown face image to the unknown_faces folder
        unknown_face_filename = os.path.join(unknown_faces_folder, f"unknown_face_{entry_time}.png")
        cv2.imwrite(unknown_face_filename, face_img)
        
        # Insert record into the unknown_faces table
        cursor.execute('''
        INSERT INTO unknown_faces (image_path, entry_time) VALUES (%s, %s)
        ''', (unknown_face_filename, entry_time))
        conn.commit()
        
        logging.info(f"Saved to unknown database: {unknown_face_filename} at {entry_time}")
    
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
    finally:
        if conn:
            conn.close()

# Function to process images for face recognition
def process_faces(face_queue):
    while True:
        try:
            if not face_queue.empty():
                file_path = face_queue.get()
                image = face_recognition.load_image_file(file_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    # Compare the detected face with the known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    name = "Unknown"
                    if matches[best_match_index]:
                        name = known_face_labels[best_match_index]
                        # Save to recognized database
                        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_to_recognized_db(name, entry_time)
                        # Record attendance
                        record_attendance(name)  # Assuming user_id is the same as name for simplicity
                    else:
                        # Save unknown face to the database and move to unknown_faces folder
                        entry_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                        save_to_unknown_db(image, entry_time)
                        # Add unauthorized alert
                        save_unauthorized_alert(f"Unknown face detected at {entry_time}")

                    # Remove the processed image
                    os.remove(file_path)
                    logging.info(f"Processed and removed: {file_path}")

        except Exception as e:
            logging.error(f"Error processing faces: {e}")

        time.sleep(1)  # Check for new images every second

# Function for face detection and saving detected faces to queue
def face_detection(face_queue):
    # Create folder for storing detected faces if it doesn't exist
    folder_path = r'C:\Users\vanch\OneDrive\Desktop\detected_faces'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Load the pre-trained DNN model for face detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # Initialize the webcam video capture
    video = cv2.VideoCapture("rtsp://192.168.0.124:554/user=admin_password=_channel=1_stream=0.sdp")

    # Variable to track the last face saved time
    last_save_time = datetime.now() - timedelta(seconds=2)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Prepare the image for the DNN model
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        current_time = datetime.now()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold for face detection
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the coordinates are within the frame dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                crop_img = frame[startY:endY, startX:endX]
                
                # Check if crop_img is not empty
                if crop_img.size == 0:
                    logging.warning("Detected face with invalid bounding box, skipping...")
                    continue
                
                ts = time.time()
                timestamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S%f")
                
                # Check if 3 seconds have passed since the last save
                if current_time - last_save_time >= timedelta(seconds=2):
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY-40), (endX, startY), (50, 50, 255), -1)
                    cv2.putText(frame, "Face", (startX, startY-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                    # Save detected face as PNG image with default size
                    face_filename = os.path.join(folder_path, f"face_{timestamp}.png")
                    cv2.imwrite(face_filename, crop_img)
                    
                    # Add the face file to the queue
                    face_queue.put(face_filename)
                    
                    # Update the last save time
                    last_save_time = current_time

        cv2.imshow("Frame", frame)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # Release resources and close windows
    video.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Create a queue for detected face images
    face_queue = queue.Queue()

    # Start the face detection thread
    face_detection_thread = threading.Thread(target=face_detection, args=(face_queue,))
    face_detection_thread.start()

    # Start the face processing thread
    face_processing_thread = threading.Thread(target=process_faces, args=(face_queue,))
    face_processing_thread.start()

    # Wait for the threads to complete
    face_detection_thread.join()
    face_processing_thread.join()
