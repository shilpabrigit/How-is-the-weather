import streamlit as st

st.markdown('<h1 style="color:black;">VGG 16 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> Cloudy, Rainy, Sunny, Sunrise_Sunset</h3>', unsafe_allow_html=True)


import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
saved_model = load_model("path_to_saved_model.h5")

# Define function to preprocess uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# Define Streamlit app
def main():
    st.title("Weather Image Classifier")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

       # Load the saved model
        saved_model_path = r"C:\Users\dell\Documents\Upskill\Project\Weather Recognition\weather_classif"
        saved_model = load_model(saved_model_path)

        # Make prediction
        prediction = saved_model.predict(img_array)
        predicted_class = np.argmax(prediction)
        class_labels = ["Cloudy", "Rainy", "Sunny", "Sunrise_Sunset"]
        predicted_label = class_labels[predicted_class]

        # Display prediction result
        st.success(f"Predicted weather: {predicted_label}")

# Run the app
if __name__ == "__main__":
    main()
