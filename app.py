import streamlit as st
from functions.collect import collect_images
from functions.train import train_model
from functions.inference import run_inference

# Load the model (if exists)
import os
import pickle

model_file = 'model.p'
if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict.get('model', None)  # Load model if it exists
        label_map = model_dict.get('label_map', {})  # Load label_map if it exists
else:
    model = None
    label_map = {}

# Set up Streamlit UI
st.title('Sign Language Recognition App')

menu = ["Collect Images", "Train Model", "Run Inference"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Collect Images":
    collect_images()

elif choice == "Train Model":
    train_model()

elif choice == "Run Inference":
    run_inference()
