import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'C:/Users/PARTH ATHALYE/Desktop/SignToText/captures'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ =[]
        y_ =[]
        img = cv2.imread(os.path.join(DATA_DIR, dir_,img_path))

        # Check if the image file is not empty
        if img is None:
            print(f"Error loading image {img_path}: Image file is empty.")
            continue
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)


# Create a dictionary to store 'data' and 'labels'
data_dict = {'data': data, 'labels': labels}

# Specify the filename for the pickle file
pickle_filename = 'data.pickle'

# Save the dictionary to a pickle file
with open(pickle_filename, 'wb') as pickle_file:
    try:
        pickle.dump(data_dict, pickle_file)
        print("Pickle file saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the pickle file: {str(e)}")
print(data_dict)