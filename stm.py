import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import os  
import tensorflow_hub as hub
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax


st.header("image class predict")

def main():
    file_upload = st.file_uploader("pilih gambar", type=['jpg','png'])
    if file_upload is not None:
        image = Image.open(file_upload)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model = tf.keras.models.load_model('model_breakhis.h5')
    shape = ((250,250,3))
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
    test_image = image.resize((250,250))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['malignant','bening']
    predictions = model.predict(test_image)
    score = tf.nn.sigmoid(predictions[0])
    score = score.numpy()
    image_class = class_name[np.argmax(score)]
    result = "the image uploaded is: {}".format(image_class)
    st.write(test_image.shape)
    return result

if __name__ == "__main__":
    main()

