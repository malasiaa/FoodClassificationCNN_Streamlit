import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras.backend as K
import pandas as pd
import base64
from keras.utils import get_file

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

# Correct raw file URL for the .h5 file on GitHub
model_url = "https://github.com/malasiaa/FoodClassificationProject_Streamlit/raw/c12f6ebb91febd443a5b41207b8b6952665f2307/vgg_foodclass.h5"

# Download the file and save it locally
local_model_path = get_file("vgg_foodclass.h5", model_url)

# Loading the model
model = tf.keras.models.load_model(local_model_path, custom_objects={"f1_m": f1_m})


#model = tf.keras.models.load_model("", custom_objects={"f1_m": f1_m})

st.set_page_config(
    page_title="Food Classification Prediction App",
    page_icon=":compuFter:",
    layout="wide",  # Use "wide" layout for a larger page width
    initial_sidebar_state="expanded",  # Expand the sidebar by default
    )

import streamlit as st

# Convert the image to bytes and then to a Base64 encoded string
def img_to_bytes(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Create an HTML string for the image
def img_to_html(img_path):
    img_bytes = img_to_bytes(img_path)
    img_html = f'<img src="data:image/gif;base64,{img_bytes}" class="img-fluid">'
    return img_html

# Use the markdown function to display the centered image
st.markdown(f'<div style="text-align: center;">{img_to_html("charlie_eatingmachine.gif")}</div>', unsafe_allow_html=True)

st.title("Food Classification App")

st.markdown("**This app supports 34 food categories:** Baked Potato, Crispy Chicken, Donut, Fries, Hot Dog, Sandwich, Taco, Taquito, Apple Pie, Burger, Butter Naan, Chai, Chapati, Cheesecake, Chicken Curry, \n"
        "Chole Bhatura, Dal Makhani, Dhokla, Fried Rice, Ice Cream, Idli, Jalebi, Kaathi Rolls, Kadai Paneer, Kulfi, Masala Dosa, Momos, Omelette, Paani Puri, Pakode, Pav Bhaji, Pizza, Samosa, and Sushi!")

# Close the container
st.markdown('')
###End of container

def import_and_predict (image_data, model):
    size=(224,224,)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)[0]
    
    return prediction

file = st.file_uploader("**Please upload a food image:**", type=["jpg","png","jpeg"])

if file is None:
    st.text("")
else:
    image=Image.open(file)
    cols = st.columns(3)
    with cols[0]:
        st.image(image, width=400, use_column_width=False)
    with cols[1]:
        predictions = import_and_predict(image, model)
        class_names = ['Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich', 'a Taco!', 'Taquitooo :)', 'Apple Pie', 'a Burguer', 'Butter Naan', 'Chai', 'Chapati', 'Cheesecake', 'Chicken Curry', 'Chole Bhatura', 'Dal Makhani', 'Dhokla', 'Fried Rice', 'Ice Cream', 'Idli', 'Jalebi', 'Kaathi Rolls', 'Kadai Paneer', 'Kulfi', 'Masala Dosa', 'Momos', 'Omelette', 'Paani Puri', 'Pakode', 'Pav Bhaji', 'Pizzaaaaa!', 'Samosa', 'Sushi! (disguised sugars)']
        string="This image most likely is: "+class_names[np.argmax(predictions)]
        st.success(string)
    with cols[2]:
        st.write(' ')