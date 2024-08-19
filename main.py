import streamlit as st
from tensorflow.keras.models import load_model   # type: ignore
from PIL import Image
import numpy as np
import os
from util import classify, set_background

# Set background image
set_background('./bgs/bg5.png')

# Set title
st.title('Pneumonia Classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('Upload a file', type=['jpeg', 'jpg', 'png'])

# Load model
model_path = 'D:/PROJECT FOLDERS/pneumonia-classification-web-app-python-streamlit-main/model/pneumonia_classifier.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"Model file not found at {model_path}")
    model = None

# Load class names
class_names_path = 'D:/PROJECT FOLDERS/pneumonia-classification-web-app-python-streamlit-main/model/labels.txt'
if os.path.exists(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = [line.strip().split(' ')[1] for line in f.readlines()]
else:
    st.error(f"Class names file not found at {class_names_path}")
    class_names = []

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    if model is not None and class_names:
        # Classify image
        class_name, conf_score = classify(image, model, class_names)

        # Write classification results
        st.write("## Prediction: {}".format(class_name))
        st.write("### Confidence Score: {:.2f}%".format(conf_score * 100))
    else:
        st.error("Model or class names are not available. Please check the files.")
