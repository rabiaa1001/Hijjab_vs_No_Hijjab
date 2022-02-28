import os
from PIL import Image
import PIL
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from main_streamlit import load_image_and_resize
import pytest

@pytest.mark.parametrize('width,height',[(150,150),(224,224),(256,256)])
def test_load_image_and_resize(width,height):
    # Ensure different sizes will be resized as expected
    img = load_image_and_resize(
        '/Users/rabia/PycharmProjects/pythonProject/hijjab_vs_non_hijjab_project/python_py_test/Test_images/hijjabs0.jpg', width, height)
    # Ensure Image is a PIL.Image.Image subclass
    assert issubclass(type(img), PIL.Image.Image)
    # Assert size of image
    assert img.size == (width, height)

@pytest.mark.parametrize('picture,image_class',[('none4.jpg','No Hijjab'),('hijjab1.jpg','Hijjab')])
def test_single_image(picture,image_class):
    # Create a class list
    CLS_LIST = ['Hijjab', 'No Hijjab']
    
    # Retrieve the model
    MODEL = load_model('hijjab_model_resnet50_v2.h5', compile=False)
    
    # Open each image to test
    img = Image.open(f'./Test_images/{picture}')
    
    # Resize the image for the model
    img = img.resize((224, 224))
    
    # Convert the image to an array
    x = image.img_to_array(img)
    
    # Reshape the image
    x = x[:, :, :3]
    
    # Assert the correct dimensions
    assert x.shape == (224, 224, 3)
    
    # Preprocess the input
    x = preprocess_input(x)
    
    # Expand the dimensions
    x = np.expand_dims(x, axis=0)
    
    # Make a prediction for the image
    pred = MODEL.predict(x)[0]
    pred_class = CLS_LIST[np.argmax(pred)]
    
    assert pred_class == image_class
