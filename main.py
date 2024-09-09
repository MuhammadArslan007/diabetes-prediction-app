import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your pre-trained model
model = tf.keras.models.load_model('VGG16_custom-v1.h5')

st.title('🩺 Diabetes Prediction App')
st.markdown(
    """
    Upload an image of a medical scan of foot or select from the sample images from sidebar to predict if the individual is diabetic or non-diabetic.
    """
)

# Sidebar - About Me
st.sidebar.header('About Me')
st.sidebar.write("**Name:** Muhammad Arslan")
st.sidebar.write("**Email:** [arslan.au189@gmail.com](mailto:arslan.au189@gmail.com)")
st.sidebar.write("**GitHub:** [MuhammadArslan007](https://github.com/MuhammadArslan007)")



st.sidebar.subheader("Sample Images")
st.sidebar.markdown("You can either upload your own image or select a sample image:")


sample_images = {
   
    "Diabetic 1": "sample_images/diabetic1.png",
    "Diabetic 2": "sample_images/diabetic2.png",
    "Non Diabetic 1": "sample_images/nondiabetic1.png",
    "Non Diabetic 2": "sample_images/nondiabetic2.png",
}

# Option to choose from the sample images or upload a file
sample_option = st.sidebar.selectbox("Choose a sample image:", list(sample_images.keys()), index=0)
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

# Load the selected sample image or uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

else:
    image = Image.open(sample_images[sample_option])
    st.sidebar.image(image, caption=f"Selected: {sample_option}", width=100)

st.image(image, caption='Selected Image for Prediction', use_column_width=False, width=200,  output_format='auto')


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


if st.button("🔍 Predict"):
 
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    decision = np.argmax(prediction)
    if decision:
        predicted_class, probability = "Diabetic", prediction[0][decision]
    else:
        predicted_class, probability = "Non Diabetic", prediction[0][decision]
    
   
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Probability:** {probability:.2f}")

