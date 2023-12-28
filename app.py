import joblib
import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNet, ResNet101
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mahotas.features import haralick

st.set_page_config(
    page_title="DermDetect",
    page_icon="🩺",
)

target_size = (224, 224)
st.title("Skin Disease Prediction")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');
        body {
            font-family: 'Poppins', sans-serif;
            background-image: url("https://img.freepik.com/free-photo/medium-shot-smiley-doctor-patient-chatting_23-2149351657.jpg?w=1060&t=st=1703710088~exp=1703710688~hmac=8aa5bbd3f1892e2d21d24ae0dafc21664dcb84b2c24b9809028367c297055441");
            background-size: cover;
            color: #333333;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin-bottom:60px;
            overflow: hidden;
        }
           .stApp {

            max-width: 700px;
            height: 690px;
            margin: 50px 0 0 20px;
             border-radius: 15px;
            background-color: rgba(255, 255, 255, 0.8);

        }
        [data-testid="stText"]{
            font-size:23px;
            font-weight:600;
        }

        [data-testid="StyledLinkIconContainer"]{
            color:#86B6F6;
            text-align:center;
            margin-top:-10px
        }
        [data-testid="block-container"]{
            overflow:hidden;

        }
        .stHeader {
            background-color: rgba(134, 182, 246, 0.8);
            color: #ffffff;
            padding: 10px;
            text-align: center;
        }
        .stSidebar {
            background-color: #86B6F6;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .stContent {

            padding: 20px;
        }
        .stTitle {
                text-align:center;
        }
        .scroll-recommendation {
            max-height: 300px;
            overflow-y: auto;
        }
        .recommendation-text {
            font-size: 22px; 
            font-weight:600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


def adjust_brightness_contrast_batch(images, alpha=1.0, beta=0):
    adjusted_images = [cv2.convertScaleAbs(image, alpha=alpha, beta=beta) for image in images]
    return adjusted_images


def extract_features(image_path):
    image_arr = np.asarray(Image.open(image_path))
    adjusted_images = adjust_brightness_contrast_batch([image_arr], alpha=1.5, beta=50)
    adjusted_image = np.array(adjusted_images[0])

    resized_image = cv2.resize(adjusted_image, target_size)
    resized_image = resized_image.reshape((1,) + target_size + (3,))

    base_model = load_model('model2.h5')
    features = base_model.predict(resized_image)

    return features


def disease_detected(features):
    ann_model = joblib.load('MobileNet_Transfer_Learning_ANN (1).joblib')
    prediction = ann_model.predict(features)
    prediction_index = int(prediction[0])
    class_labels = ['Common Nevus', 'Atypical Nevus', 'Melanoma']

    formatted_prediction = f"<div style='font-size: 30px;'>{class_labels[prediction_index]}</div>"

    return class_labels[prediction_index]


def severity_level(prediction):
    if prediction == 'Common Nevus':
        return 'Low'
    elif prediction == 'Atypical Nevus':
        return 'Moderate '
    elif prediction == 'Melanoma':
        return 'High'
    else:
        return 'Severity Unknown'


def recommendation_for_disease(predicted_disease, severity_level):

    if predicted_disease == 'Melanoma':
        if severity_level == 'High':
            return "This appears to be a high-risk melanoma. It is critical to consult a dermatologist immediately for further evaluation and treatment."
        elif severity_level == 'Moderate':
            return "This may indicate a moderate-risk melanoma. It is recommended to schedule a dermatologist appointment for a thorough examination."
        else:
            return "Melanoma detected. Consult a dermatologist for accurate diagnosis and advice."

    elif predicted_disease == 'Atypical Nevus':
        if severity_level == 'High':
            return "This atypical nevus shows high-risk features. Immediate consultation with a dermatologist is recommended for further assessment."
        elif severity_level == 'Moderate':
            return "This atypical nevus may have moderate-risk features. Schedule a dermatologist appointment for a detailed examination."
        else:
            return "Atypical nevus detected. Regular skin checks and follow-ups with a dermatologist are advised."

    elif predicted_disease == 'Common Nevus':
        return "This common nevus appears to be benign. Keep an eye on any changes, and consult a dermatologist if you notice abnormalities."

    else:
        return "Unknown disease. Consult a dermatologist for accurate diagnosis and advice."


uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:

    if save_uploaded_image(uploaded_image):

        display_image = Image.open(uploaded_image)

        extracted_features = extract_features(os.path.join('uploads', uploaded_image.name))

        prediction = disease_detected(extracted_features)

        severity = severity_level(prediction)
        recommendation = recommendation_for_disease(prediction, severity_level)

    col1, col2 = st.columns(2)

    with col1:
        st.header('Your Image')
        st.image(display_image)

    with col2:
        st.header('Predicted Disease')
        st.text(f'Name: {prediction}')
        st.text(f'Severity: {severity}')
        st.markdown(f'<div class="scroll-recommendation recommendation-text">Recommendation:{recommendation}</div>',
                    unsafe_allow_html=True)
