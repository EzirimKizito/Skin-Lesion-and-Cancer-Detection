import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the models
lesion_model_path = 'Model_for_lesion.h5'
cancer_model_path = 'Model_for_cancer.h5'

lesion_model = load_model(lesion_model_path)
cancer_model = load_model(cancer_model_path)

# Class labels for lesion model
lesion_classes = {
    'AKIEC': 'Actinic Keratoses and Intraepithelial Carcinoma / Bowen’s Disease',
    'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis-like Lesions',
    'DF': 'Dermatofibroma',
    'MEL': 'Melanoma',
    'NV': 'Nevus (a type of lesion often caused by a proliferation of melanocytes)',
    'Normal': 'Normal Skin',
    'VASC': 'Vascular Lesions'
}

cancer_classes = ['Non-Cancerous', 'Cancerous']

# Set up the app layout
st.title('Skin Lesion Diagnosis App')
st.write("By Mariam Makasaa")

st.sidebar.header('1. Input Data')
uploaded_file = st.sidebar.file_uploader("Choose a skin lesion image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_img(uploaded_file, target_size=(256, 256))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize

    st.image(image, caption='Uploaded Image.', use_column_width=False, width=200)

    st.sidebar.write("")

    # Display prediction buttons
    if st.sidebar.button('Predict Lesion Class'):
        prediction = lesion_model.predict(img_array)
        predicted_class = max(lesion_classes, key=lambda k: prediction[0][lesion_classes.keys().index(k)])
        st.sidebar.write(f"Predicted Lesion Class: {predicted_class}")
        st.sidebar.write(f"{predicted_class}: {lesion_classes[predicted_class]}")

    if st.sidebar.button('Diagnose'):
        prediction = cancer_model.predict(img_array)
        predicted_class = cancer_classes[np.argmax(prediction)]
        st.sidebar.write(f"Diagnosis: {predicted_class}")

st.sidebar.write("Upload an image and click on the respective button to get the prediction.")
