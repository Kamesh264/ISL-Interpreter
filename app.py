import numpy as np
import pandas as pd
import streamlit as st
import cv2
import pickle
import mediapipe as mp
from sklearn.preprocessing import StandardScaler



with open('trained_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('trained_models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

confidence = 0.5
mp_options = mp.solutions.hands.Options(
    max_num_hands=2,
    min_detection_confidence=confidence,
    min_tracking_confidence=0.5,  # adjust as needed
    use_gpu=False  # disable GPU for now
)
mp_hands = mp.solutions.hands;
hands = mp_hands.Hands(static_image_mode=True, options=mp_options)


# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=confidence)

df = pd.read_csv("landmark_data/Gestures_sentences.csv")
my_dict = df.set_index('gesture_names')['sentence'].to_dict()
for key in my_dict:
    print(key, ':', my_dict[key])

final_dict = {}
for key in my_dict:
    t = []
    words = key.split(',')
    for word in words:
        t.append(word)
    s = ' '.join(t)
    final_dict[s] = my_dict[key]
for key in final_dict:
    print(key, ':', final_dict[key])
    

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
# print(cap)

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) 
if not (cap.isOpened()):
    print("Could not open video device")

while run:
    ret, image = cap.read()

    if not ret:
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = hands.process(image)

    image.flags.writeable = True
    
    #converting the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    both_hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                # Extract x, y coordinates (relative to image dimensions)
                x = landmark.x
                y = landmark.y
                # Append coordinates to the list
                landmarks.append((x, y))
            both_hand_landmarks.append(landmarks)
        
        if len(both_hand_landmarks) == 1:
            both_hand_landmarks.append([(0, 0)] * len(both_hand_landmarks[0]))
        values = list(np.array(both_hand_landmarks).flatten())
        values = scaler.transform([values])
        predicted = loaded_model.predict(values)

        cv2.rectangle(image, (0,0), (160, 60), (245, 90, 16), -1)
        # Displaying Class
        cv2.putText(image, 'Predicted Gesture'
                    , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(predicted[0])
                    , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
        threshold_list.append(predicted[0])

        if threshold_list.count(predicted[0]) >= threshold:
            # Add caption text
            if seq[-1] != predicted[0]:
                caption = generate_caption(predicted[0], seq)
            if caption == '':
                caption= prev_caption
            else:
                prev_caption = caption
            threshold_list = []
    caption_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    caption_x = int((image.shape[1] - caption_size[0]) / 2)
    caption_y = image.shape[0] - 10  # Adjust 10 for padding
    cv2.putText(image, caption, (caption_x, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    
    # cv2.imshow('Sign Translator', image)
    FRAME_WINDOW.image(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
else:
    cap.release()
    cv2.destroyAllWindows()
    st.write("Stopped")