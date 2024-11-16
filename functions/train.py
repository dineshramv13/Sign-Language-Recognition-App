
import streamlit as st
import os
import cv2
import numpy as np
import pickle
import logging
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mediapipe as mp
from logs.logger import logging

# Set up logging


# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

def train_model():
    st.subheader("Train Model")
    if st.button("Train"):
        data = []
        labels = []
        label_map = {}
        label_index = 0
        expected_feature_length = 42

        start_time = time.time()
        logging.info("Training process started.")

        for dir_ in os.listdir(DATA_DIR):
            class_dir = os.path.join(DATA_DIR, dir_)
            if not os.path.isdir(class_dir):
                continue

            logging.info(f"Processing directory: {class_dir}")
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    logging.warning(f"Failed to read image: {img_file}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                try:
                    results = hands.process(img_rgb)
                except Exception as e:
                    logging.error(f"Error processing {img_file}: {e}")
                    continue

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux = []
                        x_, y_ = [], []

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                        if len(data_aux) == expected_feature_length:
                            data.append(data_aux)
                            if dir_ not in label_map:
                                label_map[dir_] = label_index
                                label_index += 1
                            labels.append(label_map[dir_])
                            logging.info(f"Processed image {img_file} in {dir_}")
                        else:
                            logging.warning(f"Inconsistent data in image {img_file}, skipping.")
                else:
                    logging.warning(f"No hands found in {img_file}.")

        # Save collected data
        with open('data.pickle', 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        logging.info("Data successfully saved to data.pickle.")

        # Load and preprocess data for model training
        data_dict = pickle.load(open('data.pickle', 'rb'))
        max_sequence_length = 42
        data = np.array([seq[:max_sequence_length] + [0]*(max_sequence_length-len(seq)) 
                         for seq in data_dict['data']], dtype='float32')
        labels = np.asarray(data_dict['labels'])
        logging.info(f"Data loaded and reshaped for training. Total samples: {len(data)}")

        # Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels)
        logging.info(f"Data split into training and test sets. Training samples: {len(x_train)}, Test samples: {len(x_test)}")

        # Hyperparameter tuning with GridSearchCV
        rf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2) 
        #if we put verbose = 0 , it will not print in terminal

        
        grid_search.fit(x_train, y_train)
        classifier = grid_search.best_estimator_
        logging.info(f"Best hyperparameters: {grid_search.best_params_}")

        # Evaluate the model
        y_predict = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_predict)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"Model accuracy: {accuracy * 100:.2f}%")

        # Classification report and confusion matrix
        report = classification_report(y_test, y_predict, target_names=label_map.keys())
        conf_matrix = confusion_matrix(y_test, y_predict)
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        
        st.text("Classification Report:")
        st.text(report)
        st.text("Confusion Matrix:")
        st.text(conf_matrix)

        # Feature importance
        feature_importances = classifier.feature_importances_
        for idx, importance in enumerate(feature_importances):
            logging.info(f"Feature {idx + 1}: {importance:.4f}")
        st.text("Feature Importances (Top 10):")
        st.text(np.round(feature_importances[:10], 4))

        # Save the trained model and label map
        with open('model.p', 'wb') as f:
            pickle.dump({'model': classifier, 'label_map': label_map}, f)
        logging.info("Model and label map saved to model.p.")

        # Record total training time
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Total training time: {total_time:.2f} seconds.")
        st.success("Model trained and saved.")
