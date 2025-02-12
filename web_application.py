import cv2
import mediapipe as mp
import copy
import itertools
import pandas as pd
import numpy as np
from tensorflow import keras
import csv
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import eventlet
import threading
import base64
import atexit # Exit for the Flask App
import time

# Setup MediaPipe Hand Detection
###########################################################
# module performs the hand recognition algorithm
mp_hands = mp.solutions.hands
# draw the detected key points
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# load the saved model ###########################################################
Model_path = r"E:\WBS DS\Final Project\All_final_project_things\Big_dataset\Mediapipe_my_dataset_ISL\ISL_app\model\keypoint_classifier_1.hdf5"

try:
    model = keras.models.load_model(Model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)



# Read the classifier labels
###########################################################
# "with" statement ensures that the file is properly closed after reading, even if an error occurs.
# encoding - use to remove weird characters
Labels_path = r"E:\WBS DS\Final Project\All_final_project_things\Big_dataset\Mediapipe_my_dataset_ISL\ISL_app\keypoint_classifier_label.csv"

try:
    with open(Labels_path, encoding='utf-8-sig') as f:
        # row[0] extracts the first column of each row.
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)] 
except FileNotFoundError:
    print(f"Error: Label file not found at {Labels_path}")
    exit(1)
except Exception as e:
    print(f"Error reading label file: {e}")
    exit(1)

# Setup Flask instance and SocketIO
###########################################################
app = Flask(__name__) # __name__ tells Flask to use the current module's name

# initializes a SocketIO instance, which enables real-time WebSocket communication for the Flask app.
# allow real-time bidirectional communication between a client (e.g., browser) and server.
socketio = SocketIO(app, cors_allowed_origins="*")
camera = None  # Global variable for the camera

# Initialize MediaPipe Hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hand gesture data processing functions
###########################################################
def get_bounding_box(image, landmarks):
    # image shape
    image_width, image_height = image.shape[1], image.shape[0]

    # Converts normalized landmarks (0-1) into actual pixel coordinates (0-image_size) and save then to numpy array
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark])

    # Find smallest box w=(xmax-xmin), h=(ymax-ymin)
    x, y, w, h = cv2.boundingRect(landmark_array)

    # return 4 corners to draw box
    return [x, y, x + w, y + h]


# calculate landmarks and convert into pixel coordinates
###########################################################
def extract_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_list = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


# preprocess the list of landmarks by converting them into relative coordinates (with respect to the first landmark) and normalizing them
###########################################################
def normalize_landmarks(landmark_list):
    base_x, base_y = landmark_list[0]

    # Convert to relative coordinates
    temp_landmark_list = [(x - base_x, y - base_y) for x, y in landmark_list]

    # The list of relative (x, y) coordinates is flattened into a single list = chain.from_iterable(['ABC', 'DEF']) → A B C D E F
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list]


# merge bounding box if both hands doing same gesture
#################################################
def merge_bounding_boxes(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return [min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)]



# Initialize camera
#################################################
camera = cv2.VideoCapture(0)


# Flask-SocketIO application that handles real-time video streaming using a web camera and MediaPipe Hands for hand tracking. It processes each frame from the camera and sends the processed image to the client over WebSockets.

# Initialize mediapipe hand model
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    # static_image_mode=True
    # min_tracking_confidence=0.5

    # SocketIO event listener
    # triggered when a client connects to the server.
    @socketio.on('connect')
    def handle_connect(): # A function that gets called when the client establishes a connection
        print("Client connected")

    @socketio.on('start_stream')
    def process_video_stream():
        """Capture frames from the camera, processes them and send them to the client."""
        global camera

        if camera is None or not isinstance(camera, cv2.VideoCapture):
            camera = cv2.VideoCapture(0)

        
        if not camera.isOpened():
            print("Error: Could not open camera.")
            socketio.emit("video_frame", {"image": "", "gesture": "Camera Error"})
            return 

        while camera.isOpened():
            success, image = camera.read()
            if not success:
                print("Failed to read frame from camera")  # Debugging message
                continue
            image = cv2.flip(image, 1)  # Flip for mirror effect

            # Process the frame using MediaPipe Hands
            # Makes the image non-writeable for better performance with MediaPipe.
            image.flags.writeable = False

            # Converts the image from BGR (OpenCV format) to RGB (MediaPipe format), as MediaPipe expects images in RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate a unique timestamp
            timestamp = int(time.time() * 1e6)  

            # Runs the image through the MediaPipe Hands model, detecting any hands in the image. The results of the hand detection (landmarks, etc.) are stored in results.
            results = hands.process(image)

            # Draw the hand annotations on the image.
            # Sets the image back to writeable after processing.
            image.flags.writeable = True

            # convert back to BGR for opencv display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Creates a deep copy of the image for debugging or other purposes (like drawing landmarks or annotations without modifying the original image).
            debug_image = copy.deepcopy(image)
            
            hand_data = []
            detected_label = None

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # Bounding box calculation
                    brect = get_bounding_box(debug_image, hand_landmarks)

                    # calculating landmarks
                    landmark_list = extract_landmarks(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = normalize_landmarks(landmark_list)

                    # Convert to dataframe
                    df = pd.DataFrame([pre_processed_landmark_list])

                    predictions = model.predict(df, verbose=0)

                    confidence = np.max(predictions)  # Get the highest confidence score

                    # np.argmax(predictions) finds the index of the maximum value in the predictions array
                    predicted_label = keypoint_classifier_labels[np.argmax(predictions)]

                    hand_data.append((brect, predicted_label, handedness.classification[0].label))


                # Chk if both hands doing the same gesture
                if len(hand_data) == 2 and hand_data[0][1] == hand_data[1][1]:
                    detected_label = hand_data[0][1]

                    # Merge bounding boxes for both hands  
                    merged_brect = merge_bounding_boxes(hand_data[0][0], hand_data[1][0])

                    # Draw Green bounding box for matched signs 
                    cv2.rectangle(debug_image, (merged_brect[0], merged_brect[1]), (merged_brect[2], merged_brect[3]), (0, 255, 0), 2)

                    # Draw landmarks for both hands
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(  
                            debug_image,  
                            hand_landmarks,  # Pass the entire hand_landmarks object, not indexed
                            mp_hands.HAND_CONNECTIONS,  
                            mp_drawing_styles.get_default_hand_landmarks_style(),  
                            mp_drawing_styles.get_default_hand_connections_style()  
                        )

                    # Draw text for the recognized sign
                    display_text = f"{detected_label} ({confidence:.2f})"  

                    cv2.putText(debug_image, display_text, (merged_brect[0], merged_brect[1] - 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                else:

                    # Draw separate red bounding boxes for different signs 
                    for i, (brect, label, handedness) in enumerate(hand_data):
                        detected_label = label

                        # draw bounding box    
                        cv2.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 255), 2)

                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            debug_image, 
                            results.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),  
                            mp_drawing_styles.get_default_hand_connections_style()
                            )

                        # Dispaly label
                        display_text = f"{detected_label} ({confidence:.2f})"

                        cv2.putText(debug_image, display_text, (brect[0], brect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Send the processed frame and detected label to frontend using Socket.IO.
            #################################################

            # Converts the debug_image into a JPEG format.
            _, buffer = cv2.imencode('.jpg', debug_image)

            # Convert the image to base64 encoding(a text-based representation of binary data). 
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            # Emit the frame and detected gesture to the frontend
            socketio.emit("video_frame", {"image": encoded_frame, "gesture": detected_label})

            # This ensures the server doesn't overload the CPU by processing frames too fast.
            # eventlet.sleep(0.03)  # Control frame rate (pauses execution for 30 milliseconds) time.sleep(0.03)
            socketio.sleep(0.02)  # Instead of eventlet.sleep(0.03)


    # disconnect
    #################################################

    @socketio.on('stop_stream')
    def stop_stream():
        global camera, hands
        if camera is not None:
            camera.release()  # Release camera
            cv2.destroyAllWindows()  # Close OpenCV windows
            camera = None
            print("Camera stopped and resources released.")
        
        # hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Reinitialize

        # Send a placeholder image instead of empty string
        with open("E:\WBS DS\Final Project\All_final_project_things\Big_dataset\Mediapipe_my_dataset_ISL\ISL_app\static\placeholder.jpg", "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')

        socketio.emit('video_frame', {'image': encoded_img, 'gesture': "Stream Stopped"})  # Clear video
    

    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")
        stop_stream()


    def cleanup():
        """Release camera resources on exit."""
        global camera
        if camera is not None and camera.isOpened():
            camera.release()
            #camera = None
        print("Camera resources released.")

    atexit.register(cleanup)

    # Flask Route (/) → When a user accesses the web app (e.g., http://localhost:5000), this function serves index.html.
    # render_template("index.html") → Loads and returns the webpage frontend (HTML file) that will display the video stream.
    @app.route("/")
    def index():
        return render_template("index.html")

    
    # Running the Web Application
    #################################################
    if __name__ == "__main__":

        # Ensures the Flask app runs only when executed directly, not when imported as a module.
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
