
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the models
lesion_model_path = '/content/drive/MyDrive/Dataset/Skin_Lesion_Datasets/All/Deployments/Model_for_lesion.h5'
cancer_model_path = '/content/drive/MyDrive/Dataset/Skin_Lesion_Datasets/All/Deployments/Model_for_cancer.h5'

lesion_model = load_model(lesion_model_path)
cancer_model = load_model(cancer_model_path)

# Class labels for lesion model
lesion_classes = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'Normal', 'VASC']
cancer_classes = ['Non-Cancerous', 'Cancerous']

st.title('Skin Lesion Diagnosis App')

# File uploader
uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_img(uploaded_file, target_size=(256, 256))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize

    st.image(image, caption='Uploaded Image.', use_column_width=False, width=200)
    st.write("")

    # Display prediction buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Predict Lesion Class'):
            prediction = lesion_model.predict(img_array)
            predicted_class = lesion_classes[np.argmax(prediction)]
            st.write(f"Predicted Lesion Class: {predicted_class}")

    with col2:
        if st.button('Diagnose'):
            prediction = cancer_model.predict(img_array)
            predicted_class = cancer_classes[np.argmax(prediction)]
            st.write(f"Diagnosis: {predicted_class}")

st.write("Upload an image and click on the respective button to get the prediction.")

