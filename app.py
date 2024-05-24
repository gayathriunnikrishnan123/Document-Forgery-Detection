import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize pixel values
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load the trained model
model = tf.keras.models.load_model('resnet_model_augmented.h5')

# Streamlit app
st.title('AADHAR CARD FORGERY DETECTION')
st.write('Upload an image for prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader('Uploaded Image')
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    class_prob = prediction[0][predicted_class]

    # Define classes
    classes = ['This is an Aadhar card', 'This is not an Aadhar card']

    # Display the prediction result
    st.subheader('Prediction Result:')
    if predicted_class == 0:
        st.success(f'Predicted class: {classes[predicted_class]}')
    else:
        st.error(f'Predicted class: {classes[predicted_class]}')
    st.write(f'Probability: {class_prob:.2f}')
