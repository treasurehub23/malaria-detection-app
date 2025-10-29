import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_models("malaria_model.h5")

st.title("Malaria Detection App")
st.write("Upload cell imge to check if it's parasitized or uninfected")

file = st.file_uploader("Selec an image", type = ["jpg", "png", "jpeg"])

if file is not None:
    img=image.load_img(file, target_size=(64,64))
    img_array= image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)/255

    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.success("The cell is uninfected")
    else:
        st.error("Cell is infected")