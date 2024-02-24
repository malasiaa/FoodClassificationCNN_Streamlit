import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras.backend as K
import pandas as pd
import base64
from keras.utils import get_file

# The training used a imbalanced dataset, hence it was used F1-Score to evaluate model performance
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load the model
from keras.utils import get_file

# URL for the .h5 file on GitHub
model_url = "https://github.com/malasiaa/FoodClassificationProject_Streamlit/raw/c12f6ebb91febd443a5b41207b8b6952665f2307/vgg_foodclass.h5"

# Download the file and save it locally
local_model_path = get_file("vgg_foodclass.h5", model_url)

# Loading the model
model = tf.keras.models.load_model(local_model_path, custom_objects={"f1_m": f1_m})

# Config page
st.set_page_config(
    page_title="Food Classification Prediction App",
    page_icon=":compuFter:",
    layout="wide",  # Use "wide" layout for a larger page width
    initial_sidebar_state="expanded",  # Expand the sidebar by default
    )

# Convert Charlie gif to bytes and then to a Base64 encoded string. Raw Base 64 code is massive, this is cleaner 
def img_to_bytes(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Create an HTML string for the gif
def img_to_html(img_path):
    img_bytes = img_to_bytes(img_path)
    img_html = f'<img src="data:image/gif;base64,{img_bytes}" class="img-fluid">'
    return img_html

# Use the markdown function to display the centered image
st.markdown(f'<div style="text-align: center;">{img_to_html("charlie_eatingmachine.gif")}</div>', unsafe_allow_html=True)

# Page Title
st.title("Food Classification App")

# Page Description
st.markdown("**This app supports 34 food categories:** Baked Potato, Crispy Chicken, Donut, Fries, Hot Dog, Sandwich, Taco, Taquito, Apple Pie, Burger, Butter Naan, Chai, Chapati, Cheesecake, Chicken Curry, \n"
        "Chole Bhatura, Dal Makhani, Dhokla, Fried Rice, Ice Cream, Idli, Jalebi, Kaathi Rolls, Kadai Paneer, Kulfi, Masala Dosa, Momos, Omelette, Paani Puri, Pakode, Pav Bhaji, Pizza, Samosa, and Sushi!")

# Blank line
st.markdown('')

def import_and_predict (image_data, model):
    '''
    This function takes an image and a pre-trained model as inputs, processes the image,
    and uses the model to make a prediction. The image is resized to the required dimensions
    for the model, converted to a NumPy array, and then fed into the model for prediction.
    The function returns the prediction result.
    '''

    # Define the target size for the image, to VGG16 model
    size=(224,224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    # Convert to a NumPy array
    img = np.asarray(image)
    # Expand the dimensions of the array to match the input shape expected by the model
    img_reshape = img[np.newaxis,...]
    # Use the model to predict the class of the image
    prediction = model.predict(img_reshape)[0]

    return prediction

# Uploader field
file = st.file_uploader("**Please upload a food image:**", type=["jpg","png","jpeg"])

# Check if a file has been uploaded
if file is None:
    st.text("")
else:
    try:
        # If a file is uploaded, open it as an image
        image=Image.open(file)
        # The page width was divided into three for no particular reason besides looks
        cols = st.columns(3)
        # Display the uploaded image in the first column with fixed width
        with cols[0]:
            st.image(image, width=400, use_column_width=False)
        # Predict the class of the uploaded image using the imported_and_predict function
        with cols[1]:
            predictions = import_and_predict(image, model)
            class_names = ['Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich', 'a Taco!', 'Taquitooo :)', 'Apple Pie', 'a Burguer', 'Butter Naan', 'Chai', 'Chapati', 'Cheesecake', 'Chicken Curry', 'Chole Bhatura', 'Dal Makhani', 'Dhokla', 'Fried Rice', 'Ice Cream', 'Idli', 'Jalebi', 'Kaathi Rolls', 'Kadai Paneer', 'Kulfi', 'Masala Dosa', 'Momos', 'Omelette', 'Paani Puri', 'Pakode', 'Pav Bhaji', 'Pizzaaaaa!', 'Samosa', 'Sushi! (disguised sugars)']
            string="This image most likely is: "+class_names[np.argmax(predictions)]
            st.success(string)
        with cols[2]:
            st.write(' ')
    except:
        st.markdown('EEUnable to categorize image, please try another one! :)\n(try by searching for the categories above in google**)')

    