import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
# Load the MNIST dataset
model = load_model('my_model.h5')
st.title('Deep Learning Model Prediction')
st.write('Upload an image to get predictions from the model.')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if st.button('Predict'):
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            st.write(f'The predicted class is: {predicted_class[0]}')
        except Exception as e:
            st.write(f'Error during prediction: {e}')
