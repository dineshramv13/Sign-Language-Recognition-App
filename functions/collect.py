
import streamlit as st
import os
import cv2
import numpy as np
import random
import logging

from datetime import datetime
from logs.logger import logging

DATA_DIR = './data'
LOG_DIR = './logs'  # Directory outside functions to store logs
os.makedirs(LOG_DIR, exist_ok=True)





def collect_images():
    st.subheader("Collect Images for Training")
    alphabet = st.text_input("Enter Alphabet to Collect", "")
    dataset_size = st.number_input("Number of images", 50, 500, 100)

    if alphabet and st.button("Start Collecting"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        class_dir = os.path.join(DATA_DIR, alphabet)
        try:
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                logging.info(f"Directory created for {alphabet}: {class_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory: {e}")
            st.error(f"Error creating directory for {alphabet}")
            return

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Frame capture failed.")
                st.warning("Error capturing frame.")
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Collecting {alphabet}, Image {counter}/{dataset_size}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")

            # Apply augmentations
            try:
                augmented_images = augment_image(frame)
            except Exception as e:
                logging.error(f"Augmentation error: {e}")
                st.error("Error during augmentation.")
                continue

            for aug_img in augmented_images:
                try:
                    cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), aug_img)
                    logging.info(f"Saved image {counter} for alphabet {alphabet}")
                    counter += 1
                    if counter >= dataset_size:
                        break
                except Exception as e:
                    logging.error(f"Failed to save image {counter}: {e}")

        cap.release()
        st.success(f"Collected {dataset_size} images for {alphabet}.")
        logging.info(f"Image collection complete for {alphabet}, total images: {dataset_size}")

def augment_image(image):
    """Apply various data augmentation techniques to the input image."""
    augmented_images = [image]  # Start with the original image

    try:
        # Horizontal flip
        augmented_images.append(cv2.flip(image, 1))

        # Rotation at different angles
        for angle in [-10, 10, 15, -15]:
            augmented_images.append(rotate_image(image, angle))

        # Brightness adjustment
        for factor in [0.6, 1.4]:  # Darker and brighter versions
            augmented_images.append(adjust_brightness(image, factor))

        # Scaling
        for scale_factor in [0.9, 1.1]:  # Slight scaling adjustments
            augmented_images.append(scale_image(image, scale_factor))

        # Light source simulation
        augmented_images.append(apply_lighting_effects(image))
    except Exception as e:
        logging.error(f"Error in augmenting image: {e}")
        raise e

    return augmented_images

def rotate_image(image, angle):
    """Rotate the image by a specified angle."""
    try:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    except Exception as e:
        logging.error(f"Failed to rotate image: {e}")
        raise e

def adjust_brightness(image, factor):
    """Adjust the brightness of the image."""
    try:
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    except Exception as e:
        logging.error(f"Failed to adjust brightness: {e}")
        raise e

def scale_image(image, scale_factor):
    """Scale the image by a specified factor."""
    try:
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    except Exception as e:
        logging.error(f"Failed to scale image: {e}")
        raise e

def apply_lighting_effects(image):
    """Simulate different lighting conditions."""
    try:
        overlay = np.ones(image.shape, dtype='uint8') * 50
        dark_image = cv2.subtract(image, overlay)
        bright_image = cv2.add(image, overlay)
        return random.choice([dark_image, bright_image])
    except Exception as e:
        logging.error(f"Failed to apply lighting effects: {e}")
        raise e
