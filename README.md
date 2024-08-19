
# Pneumonia Classification Web App!
![Screenshot (5)](https://github.com/user-attachments/assets/2d6451a5-602b-49a2-9e7e-9eab0afd53d2)


This Streamlit web application allows users to classify chest X-ray images to detect pneumonia. The app uses a pre-trained machine learning model to predict whether an uploaded X-ray image shows signs of pneumonia.

## Features

- **Background Image**: A custom background image is set for the app using Streamlit's HTML capabilities.
- **Image Upload**: Users can upload a chest X-ray image in JPEG or PNG format.
- **Image Classification**: The app uses a trained TensorFlow model to classify the uploaded X-ray image as either showing signs of pneumonia or being normal.
- **Confidence Score**: Along with the prediction, the app displays a confidence score indicating how sure the model is about its prediction.

## Prerequisites

- Python 3.7 or above
- Streamlit
- TensorFlow
- Pillow

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/pneumonia-classification.git
    cd pneumonia-classification
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the pre-trained model and labels:

    Place the model (`pneumonia_classifier.h5`) and labels (`labels.txt`) files in the `model/` directory.

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open the app in your browser:

    Streamlit will automatically open the app in your default browser. If not, navigate to `http://localhost:8501`.

4. Upload a chest X-ray image:

    Upload a JPEG or PNG image of a chest X-ray for classification.

5. View the results:

    The app will display the uploaded image along with the predicted class (e.g., "Pneumonia" or "Normal") and a confidence score.

## Files

- `app.py`: The main Streamlit app code.
- `util.py`: Utility functions for setting the background and classifying images.
- `model/pneumonia_classifier.h5`: Pre-trained TensorFlow model for pneumonia classification.
- `model/labels.txt`: Labels for the classification model.
- `bgs/bg5.png`: Background image for the app.
- 
## Acknowledgments

- This project was inspired by the need for efficient pneumonia detection using deep learning.
- The model was trained using the [Chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) available on Kaggle.


