import os
from PIL import Image
import PIL
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from main_streamlit import load_image_and_resize




def test_load_image_and_resize():

    img = load_image_and_resize('/hijjab_vs_non_hijjab_project/python_py_test/Test_images/images/hijjabs0.jpg', 224, 224)
    # Ensure Image is a PIL.Image.Image subclass
    assert issubclass(type(img),PIL.Image.Image)
    # Assert size of image
    assert img.size == (224,224)

def test_single_image():
    CLS_LIST = ['Hijjab', 'No Hijjab']
    MODEL = load_model('/Users/rabia/PycharmProjects/pythonProject/hijjab_vs_non_hijjab_project/python_py_test/hijjab_model_resnet50_v2.h5',compile=False)
    img = Image.open('/hijjab_vs_non_hijjab_project/python_py_test/Test_images/images/none4.jpg')
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = x[:, :, :3]
    assert x.shape == (224,224,3)

    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = MODEL.predict(x)[0]
    pred_class = CLS_LIST[np.argmax(pred)]
    assert pred_class == 'No Hijjab'
