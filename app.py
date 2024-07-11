import numpy as np
import pandas as pd
import streamlit as st
import cv2
import pickle
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# Load pre-trained models (assuming they're in 'trained_models' folder)
with open('trained_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('trained_models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Create MediaPipe Hands options (GPU disabled for now)
mp_options = mp.solutions.hands.Hands.Options(
    max_num_hands=2,
    min_detection_confidence=confidence,
    min_tracking_confidence=0.5,  # adjust as needed
    use_gpu=False  # Disable GPU for now (consider enabling if supported)
)

# Create MediaPipe Hands instance
hands = mp_hands.Hands(static_image_mode=True, options=mp_options)

# Define dictionary for gestures and sentences
df = pd.read_csv("landmark_data/Gestures_sentences.csv")
my_dict = df.set_index('gesture_names')['sentence'].to_dict()
final_dict = {}
for key in my_dict:
    words = key.split(',')
    s = ' '.join(words)
    final_dict[s] = my_dict[key]

word_limit = 3

def generate_caption(word, seq):
    res = ''
    if len(seq) < word_limit:
        seq.append(word)
        seq.append(word)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]

    elif len(seq) == word_limit:
        seq.pop(0)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]
        seq.append(word)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]

    return res

threshold_list = []
threshold = 20
seq = ['None']
caption = ''
prev_caption = ''

# Streamlit App
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Consider removing DSHOW if not necessary

while run:
    ret, image = cap.read()

    if not ret:
        continue

    # Convert image to RGB, set flags
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process image with MediaPipe Hands
    results = hands.process(image)

    # Restore image flags and convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    both_hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract and append hand landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                landmarks.append((x, y))
            both_hand_landmarks.append(landmarks)

        # Handle single hand scenario (append empty list)
        if len(both_hand_landmarks) == 1:
            both_hand_landmarks.append([(0, 0)] * len(both_hand_landmarks[0]))

        # Flatten and scale landmarks
        values = list(np.array(both_hand_landmarks).flatten())
        values = scaler.transform([values])

        # Predict gesture
        predicted = loaded_model.predict(values)

        # Draw prediction rectangle
        cv2.rectangle(image, (0, 0), (160, 60), (245, 90, 16), -1)
        cv2.putText(image, 'Predicted Gesture', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(predicted[0]), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Update threshold list and caption
        threshold_list.append(predicted[0])
        if threshold_list.count(predicted[0]) >= threshold:
            if seq[-1] != predicted[0]:
                caption = generate_caption(predicted[0], seq)
            if caption == '':
                caption = prev_caption
            else:
                prev_caption = caption
            threshold_list = []

        # Display caption with proper positioning
        caption_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        caption_x = int((image.shape[1] - caption_size[0]) / 2)
        caption_y = image.shape[0] - 10  # Adjust 10 for padding
        cv2.putText(image, caption, (caption_x, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display image on Streamlit
    FRAME_WINDOW.image(image)

    # Handle keyboard input for quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when finished
cap.release()
cv2.destroyAllWindows()
st.write("Stopped")
