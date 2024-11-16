# Sign Language Recognition 

This project is a machine learning application for recognizing American Sign Language (ASL) letters. It utilizes **MediaPipe Hands** for detecting hand landmarks and a **Random Forest Classifier** for recognizing letters. The project also supports **real-time predictions**, allowing users to build complete sentences by recognizing individual letters and tracking previously signed words.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [How It Works](#how-it-works)  
- [Project Structure](#project-structure)  
- [Technologies Used](#technologies-used)  
- [How to Run](#how-to-run)  
- [Future Improvements](#future-improvements)  

---

## Overview

Sign language is a vital communication tool for the hearing- and speech-impaired. This project bridges the gap by recognizing ASL letters and forming sentences in real-time. It processes hand gestures captured through a webcam or pre-recorded images and translates them into meaningful words and sentences.

---

## Features

1. **Real-Time Prediction**:  
   - Predicts the "current word" being signed through live webcam input.  
   - Keeps track of previously signed words, enabling sentence formation.

2. **Letter Recognition**:  
   Recognizes individual ASL letters from static images or real-time video frames.

3. **Streamlit Interface**:  
   An interactive and user-friendly interface for training the model and testing predictions.

4. **Data Storage and Model Reusability**:  
   Saves preprocessed data and the trained model for future use, eliminating the need for retraining every session.

5. **Hand Landmark Detection**:  
   - Uses **MediaPipe Hands** to extract 21 key hand landmarks.  
   - Normalizes features for scale and position invariance.

---

## How It Works

1. **Data Collection**:  
   - Images of hands representing ASL letters are organized in a folder structure where each subfolder corresponds to a letter.  
   - The images are processed with **MediaPipe Hands** to extract 42 features (21 landmarks, each with X and Y coordinates).  

2. **Training the Model**:  
   - Features are used to train a **Random Forest Classifier**, a robust machine learning model.  
   - The trained model and a label map are saved for reuse.

3. **Real-Time Prediction Pipeline**:  
   - A webcam captures hand gestures in real time.  
   - **MediaPipe Hands** extracts hand landmarks.  
   - The model predicts the letter based on the features.  

4. **Sentence Formation**:  
   - The predicted letter is displayed as the "current word."  
   - Previously recognized letters are stored and displayed to form complete sentences.  

5. **Interactive GUI**:  
   - Users can train the model, test it on images, or switch to real-time mode using the **Streamlit** application.  
   - Real-time predictions are displayed alongside the webcam feed.

---

## Project Structure

```plaintext
.
├── data/
│   ├── A/         
│   ├── B/         
│   └── ...        
├── app.py         
├── model.p        
├── data.pickle    
├── logs/          
│   └── logger.py  
└── README.md      
```


## Technologies Used

- **Python**: Core programming language  
- **Streamlit**: Interactive GUI for model training, testing, and real-time prediction  
- **MediaPipe Hands**: Hand landmark detection  
- **Scikit-learn**: Machine learning library for training and evaluating the Random Forest model  
- **OpenCV**: Captures webcam input and processes images  
- **NumPy**: Efficient data manipulation  

---

## How to Run


**Clone the repository:**

   ```bash
   git clone https://github.com/dineshramv13/Sign-Language-Recognition-App
   ```

**Install the required dependencies:**
```bash
pip install -r requirements.txt
```









### Running the Application

1. **Start the Streamlit Application**:  
   ```bash
   streamlit run app.py
   ```

2. **Train the Model**:  
   - Place your dataset in the `data/` folder, with subfolders named after the letters they contain.  
   - Open the Streamlit app and click "Train Model" to start the training process.  

3. **Test Real-Time Predictions**:  
   - Ensure your webcam is connected.  
   - Switch to the "Real-Time Prediction" mode in the Streamlit app.  
   - Perform ASL gestures in front of the webcam to see real-time predictions of the "current word" and previously recognized words.  

---

## Future Improvements

1. **Integration with Natural Language Processing (NLP)**:  
   Enable grammar correction and automatic sentence structuring.  

2. **Expand Dataset**:  
   Include dynamic gestures and more classes like words or phrases.  

3. **Advanced Deep Learning Models**:  
   Experiment with CNNs or LSTMs for improved accuracy on complex gestures.  

4. **Mobile App Deployment**:  
   Build a mobile application for portable sign language recognition.  


