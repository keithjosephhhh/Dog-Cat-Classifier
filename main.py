import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

IMAGE_DIM = (100, 100)

st.set_page_config(page_title="Home", layout="centered")
st.title("This is a Dog/Cat Classifier")

st.markdown("---")
st.subheader("Upload a Dog or Cat picture to see the predictions!")

uploaded_file = st.file_uploader(label="", type=["jpg", "png"])
st.markdown("---")

model = load_model('Class.h5')

def resize_image(img):
    img_arr = img_to_array(img)
    img_tensor = tf.image.resize(img_arr, IMAGE_DIM)
    img_arr = img_tensor.numpy()
    return img_arr


def predict(img):
    img_arr = resize_image(img)
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    img_arr = img_arr.astype('float16')
    img_arr /= 255.0

    pred = model.predict(img_arr)
    return pred

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    pred = predict(img)

    if pred[0][0] > 0.55:
        st.subheader("Predicted: It is a Dog!")
    else:
        st.subheader("Predicted: It is a Cat!")

    st.image(img, use_column_width=True)