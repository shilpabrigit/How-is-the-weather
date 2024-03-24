
import streamlit as st
st.title("Weather Recognition model using VGG16")
st.write("The following model classified the uploaded images into Cloudy, Rainy,Sunny, Sunrise/Sunset")

import tensorflow
import keras
import numpy

#upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image 
if uploaded_file is not None:
   st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
