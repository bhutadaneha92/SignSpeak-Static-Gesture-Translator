# SignSpeak-Static-Gesture-Translator

## Summary

SignSpeak: Static-Gesture-Translator is a real-time sign language detection web application designed to recognize static hand gestures and convert them into text. 
The primary aim of this project is to facilitate communication for individuals with hearing or speech impairments by translating Indian Sign Language (ISL) gestures into text in real time.

## Features of Web Application

Sidebar: Displays common hand gestures for reference.

Real-time Video Stream: Captures video input and detects hand gestures dynamically.

Socket.IO Integration: Enables seamless real-time communication between frontend and backend.

Frontend requests video stream → Backend processes frames.

Backend sends processed frames → Frontend updates the UI dynamically.

<img width="887" alt="SignSpeak" src="https://github.com/user-attachments/assets/ef749e25-d59f-4e78-b772-1eb308da6919" />


## Languages and Libraries Used

### Frontend
  TypeScript
  JavaScript
  HTML
  CSS

### Backend
  Python
  Flask-SocketIO
  AI/ML
  MediaPipe
  TensorFlow
  OpenCV

## Dataset used for this project

  Citation: Tyagi, Akansha; Bansal, Sandhya (2022), “Indian Sign Language-Real-life Words”, Mendeley Data, V2, doi: 10.17632/s6kgb6r3ss.2

  Download Link: Mendeley Dataset

## Key Learnings

Implementing real-time image processing using OpenCV and MediaPipe.

Utilizing TensorFlow for sign language classification.

Establishing WebSocket connections with Flask-SocketIO for real-time streaming.

Creating a user-friendly interface with interactive components.

Challenges Overcame

Uniform confidence scores issue: Initially, the model returned uniform confidence scores for different images. This was resolved by adjusting hyperparameters and improving dataset augmentation.

Real-time processing latency: Optimized model inference and frame transmission to ensure smooth real-time detection.

Frontend-Backend Synchronization: Ensured a seamless connection between video streaming, gesture detection, and UI updates.

Additional Reflections

Developing this project reinforced the importance of real-time optimization in AI applications. Future enhancements could include:

Support for dynamic gestures.

Expanding the dataset for improved accuracy.

Integrating text-to-speech conversion for better accessibility.


![image](https://github.com/user-attachments/assets/d483b497-d20f-450e-9a59-064af76175aa)
