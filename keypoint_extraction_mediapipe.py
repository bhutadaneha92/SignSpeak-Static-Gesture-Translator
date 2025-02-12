import cv2
import mediapipe as mp
import csv
import copy
import itertools
import string
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Create a directory to save annotated images
# annotated_dir = r'E:\WBS DS\Final Project\All_final_project_things\Big_dataset\Annotated_Images'
# os.makedirs(annotated_dir, exist_ok=True)

# Functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    # map(function, iterable) applies function to each element in iterable and returns an iterator.
    max_value = max(list(map(abs, temp_landmark_list)))

    return [n / max_value for n in temp_landmark_list] # list comprehension

def logging_csv(label, landmark_list):
    csv_path = r'Big_dataset\Mediapipe_my_dataset_ISL\keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

# Define image folder path
train_folder_path = r'E:\WBS DS\Final Project\All_final_project_things\Big_dataset\Mediapipe_my_dataset_ISL\Mediapipe_dataset_ISL_real_life_12_words'

with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    # labels are nothing but class_folder names
    for class_folder in os.listdir(train_folder_path):  # Iterate over class folders (A, B, C, etc.)
        class_folder_path = os.path.join(train_folder_path, class_folder)

        if not os.path.isdir(class_folder_path):  # Skip non-folder files
            continue

        # List all image files in the class folder
        IMAGE_FILES = [os.path.join(class_folder_path, img) for img in os.listdir(class_folder_path) if img.endswith('.jpg')]

        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis (Horizontal) for correct handedness output 
            image = cv2.flip(cv2.imread(file), 1)

            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            # print('Handedness:', results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue

            annotated_image = image.copy()
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                # results.multi_hand_landmarks return [x,y,z] coordinates
                # Landmark calculation
                landmark_list = calc_landmark_list(annotated_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Save keypoints with the corresponding class label (A, B, C, etc.)
                logging_csv(class_folder, pre_processed_landmark_list)

                # Draw landmarks on the image
                # mp_drawing.draw_landmarks(
                #     annotated_image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                #)

            # Save annotated image
            #save_path = os.path.join(annotated_dir, f"{class_folder}_annotated_{idx}.png")
            #cv2.imwrite(save_path, annotated_image)

print("Processing completed! Keypoint of images saved in 'keypoint.csv' file.")