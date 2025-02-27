# SignSpeak: Static Gesture Translator

## Summary

SignSpeak: Static-Gesture-Translator is a real-time sign language detection web application designed to recognize static hand gestures and convert them into text. 
The primary aim of this project is to facilitate communication for individuals with hearing or speech impairments by translating Indian Sign Language (ISL) gestures into text in real time.

# Flow of Project

  1. I used the Mediapipe library to extract the landmarks from the hand of the user.
  2. Mediapipe uses RGB images with multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. 
  3. A hand landmark model that operates on the cropped image region defined by the palm detector and returns 3D hand keypoints.
  4. The landmarks are then fed into a neural network for training to classifies the gestures based on the detected keypoints.

![image](https://github.com/user-attachments/assets/d483b497-d20f-450e-9a59-064af76175aa)


## Features of Web Application

  1. Sidebar: Displays common hand gestures for reference
  2. Real-time Video Stream: Captures video input and detects hand gestures dynamically.
  3. Socket.IO Integration: Enables seamless real-time communication between frontend and backend.
  4. Frontend requests video stream → Backend processes frames.
  5. Backend sends processed frames → Frontend updates the UI dynamically.

<img width="887" alt="SignSpeak" src="https://github.com/user-attachments/assets/ef749e25-d59f-4e78-b772-1eb308da6919" />

[SignSpeak.webm](https://github.com/user-attachments/assets/6b803dbf-87fd-4952-9a40-e3550240530d)


## Languages and Libraries Used

### Frontend
  TypeScript, JavaScript, HTML, CSS

### Backend
  Python, Flask-SocketIO, AI/ML, MediaPipe, TensorFlow, OpenCV

## Key Learnings

  1. Implementing real-time image processing using OpenCV and MediaPipe.
  2. Utilizing TensorFlow for sign language classification.
  3. Establishing WebSocket connections with Flask-SocketIO for real-time streaming.
  4. Creating a user-friendly interface with interactive components.

## Future enhancements could include:

    1. Expand the dataset by including more examples of ISL gestures and adding multiple sign languages from various countries.
    2. Text-to-Speech Integration: Implement a feature to convert recognized gestures into spoken words
    3. Dynamic Gesture Recognition: Extend the model to recognize dynamic gestures for more complex sign language expressions.

## Dataset used for this project

  Download Link: [Mendeley Dataset](https://data.mendeley.com/datasets/s6kgb6r3ss/2)
  Citation: Tyagi, Akansha; Bansal, Sandhya (2022), “Indian Sign Language-Real-life Words”, Mendeley Data, V2, doi: 10.17632/s6kgb6r3ss.2
