# DermDetect 🩺

DermDetect is a web application that leverages deep learning to predict skin diseases from uploaded images. It provides predictions along with severity levels and recommendations for further action.<br>
link to website:[DermDetect](https://dermdetect16.onrender.com/)
## Table of Contents 📋

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [File Structure](#file-structure)
- [License](#license)
- [Contributing](#contributing)

## Overview 🌟

DermDetect is designed to assist in the preliminary screening of skin diseases by analyzing images of skin lesions. It uses a pre-trained deep learning model to classify the skin condition into three categories: Common Nevus, Atypical Nevus, and Melanoma. The app also provides a severity level and recommendations based on the prediction.

## Features ✨

- 📸 Upload an image of a skin lesion.
- 🌞 Automatic adjustment of image brightness and contrast.
- 🔍 Prediction of skin disease from the uploaded image.
- 📝 Display of predicted disease name and severity level.
- 📋 Recommendations based on the predicted disease.

## Installation 🛠️

### Prerequisites

- Python 3.7+
- Streamlit
- TensorFlow
- OpenCV
- Pillow
- NumPy
- joblib

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/DermDetect.git
    cd DermDetect
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download and place the models (`model2.h5` and `MobileNet_Transfer_Learning_ANN.joblib`) in the project directory.

## Usage 🚀

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your browser and navigate to `http://localhost:8501`.

3. Upload an image of a skin lesion to get the prediction, severity level, and recommendations.

## Technologies 🧪

- **Frontend:** Streamlit
- **Backend:** TensorFlow, OpenCV, NumPy, Pillow
- **Models:** Pre-trained CNN model and an ANN model for classification

## File Structure 📁

```plaintext
DermDetect/
│
├── app.py                   # Main application script
├── requirements.txt         # Python package dependencies
├── model2.h5                # Pre-trained CNN model for feature extraction
├── MobileNet_Transfer_Learning_ANN.joblib  # ANN model for classification
├── uploads/                 # Directory for storing uploaded images
└── README.md                # Readme file
