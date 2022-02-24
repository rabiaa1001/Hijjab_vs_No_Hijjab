import os
import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st

# Module level constants
MODEL = '/Hijjab_vs_No_Hijjab/model/hijjab_model_resnet50_v1.h5'
DISPLAY_IMAGE_WIDTH,DISPLAY_IMAGE_HEIGHT = 300,300
IMAGE_WIDTH,IMAGE_HEIGHT = 224,224
CLS_LIST = ['Hijjab', 'No hijjab']


def load_image(image_file:str,width:int,height:int):
    """
    Load the image and resize it according to params
    Params: image_file:str,width:int,height:int
    Return Resized PIL image
    """
    img = Image.open(image_file)
    img = img.resize((width,height))
    return img


def predict_single_image(single_image:str):
    """
    Given a single image, predic whether or not the woman is wearing hijjab
    Params: single_image:str
    Return Display the class of the image to Streamlit
    """
    model = load_model(MODEL)

    x = image.img_to_array(single_image)
    x = x[:,:,:3]

    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Model Predictions
    pred = model.predict(x)[0]

    st.markdown("<h4 style='text-align: left; color: gray;'>Model Identified this woman as having : </h4>",
        unsafe_allow_html=True)

    # result = CLS_LIST[np.argmax(pred)]

    # Just for a nice display, pink for hijjab, green for no hijjab
    if np.argmax(pred) == 0:
        st.markdown(
            "<h5 style='text-align: left; color: pink;'>Hijjab </h5>",
            unsafe_allow_html=True)
    elif np.argmax(pred) == 1:
        st.markdown(
            "<h5 style='text-align: left; color: green;'>No Hijjab </h5>",
            unsafe_allow_html=True)


def main():
    """
    Image Augmentation for small datasets
    Params: curr_path:str,new_path:str,num_of_images_to_create:str
    Return Augmented images in new folder
    """
    st.markdown("<h1 style='text-align: center; color: black;'>Hijjab or No Hijjab??? </h1>",
                unsafe_allow_html=True)

    # Banner pic
    st.image(os.path.join('./web_images/hijjab_banner.png'), use_column_width='auto')

    # Sidebar Image and Texts
    st.sidebar.image(os.path.join('./web_images/together.jpg'), use_column_width=True)
    st.sidebar.markdown("<h1 style='text-align: left; color: pink;'>"
                        "Upload a single or small batch of images</h1>",
                         unsafe_allow_html=True)
    st.sidebar.markdown("<h4 style='text-align: center; color: darkgray;'>"
                        "Please ensure the image is a picture of a woman, "
                        "either with or without a hijjab on </h4>",
                        unsafe_allow_html=True)

    # Upload Image to sidebar
    uploaded_image = st.sidebar.file_uploader('Upload your portrait here', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:

        # View Uploaded Image
        st.image(load_image(uploaded_image,
                DISPLAY_IMAGE_WIDTH,
                DISPLAY_IMAGE_HEIGHT,
                ),output_format= 'JPEG')

        # View details
        file_details = {"filename": uploaded_image.name, "filetype": uploaded_image.type,
                        "filesize": uploaded_image.size}

        st.write(file_details)

        # Send image to test_on_single_image function to predict whether or not the woman is wearing a hijjab
        predict_single_image(load_image(uploaded_image,IMAGE_WIDTH,IMAGE_HEIGHT))


if __name__ == '__main__':
    main()

