# Import necessary libraries
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
import warnings
import pandas as pd
import os
import keras
import tf_keras as k3

warnings.filterwarnings("ignore")

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Car Detection Application",
    page_icon="🚗",  
)

IMG_SIZE = 224


def predict_image(image_arr, img_size, model, class_map):
    """Predict bounding box and car model class."""
    if image_arr.shape[-1] in [1, 2, 4]:
        image_arr = tf.image.grayscale_to_rgb(image_arr)

    image = tf.image.resize(image_arr, [img_size, img_size]).numpy()
    image = np.expand_dims(image, axis=0)

    bbox, class_info = model.predict(image)
    output_class_name = class_map[np.argmax(class_info)]
    class_confidence = class_info[0].max()

    x, y, w, h = bbox[0]
    height_factor = image_arr.shape[0] / IMG_SIZE
    width_factor = image_arr.shape[1] / IMG_SIZE

    x, w = x * width_factor, w * width_factor
    y, h = y * height_factor, h * height_factor

    # Create a dataframe with just the highest confidence class
    df = pd.DataFrame({
        'Car Name': [output_class_name],
        'Confidence Level': [class_confidence]
    })

    return output_class_name, df, class_confidence, np.array([x, y, w, h]).astype(int)


def image_with_bbox(image_arr, bounding_box):
    """Draw bounding box on the image."""
    x, y, w, h = bounding_box
    image_arr = cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return image_arr


def IOU(y_true, y_pred):
    """Calculate Intersection Over Union (IOU)."""
    x1, y1, w, h = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x1_pred, y1_pred, w_pred, h_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    diff_width = np.minimum(x1 + w, x1_pred + w_pred) - np.maximum(x1, x1_pred)
    diff_height = np.minimum(y1 + h, y1_pred + h_pred) - np.maximum(y1, y1_pred)
    intersection = diff_width * diff_height

    area_true = w * h
    area_pred = w_pred * h_pred
    union = area_true + area_pred - intersection

    intersection_sum = 0
    union_sum = 0
    for j, _ in enumerate(union):
        if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
            intersection_sum += intersection[j]
            union_sum += union[j]

    iou = np.round(intersection_sum / (union_sum + tf.keras.backend.epsilon()), 4)
    return iou.astype(np.float32)


def IoU(y_true, y_pred):
    return tf.py_function(IOU, [y_true, y_pred], Tout=tf.float32)


# Use st.cache_resource for caching the model
@st.cache_resource
def load_the_pretrained_model():
    model = k3.models.load_model(
        './model/mobile_net_combi_with_aug_1_10-acc_0.8687-IoU_0.8758',
        custom_objects={'IoU': IoU}
    )
    
    with open('./model/class_names.pickle', 'rb') as f:
        class_names = pickle.load(f)
    
    return model, class_names


with st.spinner(text="Loading Model into the memory..."):
    model, class_names = load_the_pretrained_model()

st.markdown('## Car Image Classification')

with st.expander("ℹ️ - About this app", expanded=False):
    st.write(
        """
- This model was trained with 8000+ images from 196 car classes. 
- It predicts the car model and localizes the car with a bounding box.
- Trained with MobileNet as the base feature extractor with two branches: one for classification and one for localization.
"""
    )
st.write("\n")

selected_image = st.sidebar.selectbox('Sample images', os.listdir('./images'))
upload_image = st.file_uploader('Upload an image containing a car', accept_multiple_files=False)

if upload_image is not None:
    uploaded_image = cv2.imdecode(np.frombuffer(upload_image.getvalue(), np.uint8), -1)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
    msg = 'Making predictions for **uploaded** image'
else:
    selected_image_link = './images/' + selected_image
    uploaded_image = cv2.imread(selected_image_link)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
    msg = 'Making predictions for **selected** image from side bar'

if st.button('Predict'):
    st.markdown(msg)
    with st.spinner(text="Predicting..."):
        car_name, df, class_confidence, bbox = predict_image(uploaded_image, IMG_SIZE, model, class_names)
        pred_image = image_with_bbox(uploaded_image, bbox)
        uploaded_image_name = upload_image.name if upload_image is not None else selected_image
        h, w, _ = uploaded_image.shape
        code = {'class': car_name, 'confidence': round(class_confidence, 3),
                'bbox': dict(zip(['x', 'y', 'w', 'h'], list(bbox)))}

    st.write("\n")

    st.markdown('#### Predicted image')
    st.json(code)
    st.image(pred_image, use_container_width=True, caption=f'{uploaded_image_name} (height-{h} & width-{w})')

    st.write("\n")
    # Display the highest-confidence prediction
    st.subheader('Car Model')
    car_name_html = f"""<a href="https://www.google.com/search?q={df['Car Name'][0].replace(' ', '+')}" 
                     target="_blank" 
                     rel="noopener noreferrer">{df['Car Name'][0]}</a>"""

    st.markdown(f"**Car Name**: {car_name_html}", unsafe_allow_html=True)
    st.markdown(f"**Confidence Level**: {df['Confidence Level'][0]:.3f}")