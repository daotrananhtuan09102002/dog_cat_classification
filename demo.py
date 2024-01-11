import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

from keras.preprocessing.image import img_to_array
from PIL import Image


def upload_image(img_file):
    if img_file is not None:
        img = Image.open(img_file)
        img = img.resize((224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        st.image(img_file, caption="Uploaded Image", use_column_width=True)
        return img
    return None


if __name__ == "__main__":
    model = utils.load_model((224, 224, 3), 2, "./model.h5")

    st.sidebar.title("Demo")
    st.sidebar.subheader("Step 1: Upload your image")
    img_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    img = upload_image(img_file)
    button = st.sidebar.button("Classify")

    if button:
        prediction = model.predict(img, verbose=0)
        pred_class = np.argmax(prediction)
        if pred_class == 0:
            st.subheader("Prediction: Cat")
        else:
            st.subheader("Prediction: Dog")
        





