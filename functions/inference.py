import streamlit as st
import os
import cv2
import numpy as np
import pickle
import time
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def run_inference():
    st.subheader("Run Real-time Inference")
    
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    label_map = model_dict['label_map']

    if model is None or not label_map:
        st.warning("Model not found or label_map missing. Please train the model first.")
        return

    reverse_label_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    words = []  
    current_word = []  
    used_letters = set()  

    last_predicted_character = None
    delay_between_predictions = 5  
    last_prediction_time = 0  

    def end_current_word():
        nonlocal current_word, used_letters
        if current_word:
            words.append(''.join(current_word))
            current_word = []
            used_letters = set() 

    current_word_placeholder = st.empty()
    words_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()

        if results.multi_hand_landmarks and (current_time - last_prediction_time) > delay_between_predictions:
            data_aux = []
            x_, y_ = [], []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = reverse_label_map.get(prediction[0], 'Unknown')

            if predicted_character not in current_word and predicted_character != last_predicted_character:
                if predicted_character.isalpha(): 
                    current_word.append(predicted_character)
                    used_letters.add(predicted_character)
                    last_predicted_character = predicted_character
                    last_prediction_time = current_time  

        current_word_placeholder.text(f"Current Word: {''.join(current_word)}")
        words_placeholder.text(f"Words: {' '.join(words)}")

        if not results.multi_hand_landmarks:
            if current_word:
                end_current_word()

        stframe.image(frame, channels="BGR")

    cap.release()
