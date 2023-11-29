import streamlit as st
import streamlit.components.v1 as components
from lime_explainer import explainer, tokenizer, METHODS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import lime
from lime import lime_image
from lime import submodular_pick

from skimage.segmentation import mark_boundaries


#def format_dropdown_labels(val):
    #return METHODS[val]['name']

# Build app
title_text = 'AI Explainability Dashboard: Image Classification Models for User uploaded Models and Data'
#subheader_text = '''1: Strongly Negative &nbsp 2: Weakly Negative &nbsp  3: Neutral &nbsp  4: Weakly Positive &nbsp  5: Strongly Positive'''

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
#st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
st.text("")
#input_text = st.text_input('Enter your text:', "")
#n_samples = st.text_input('Number of samples to generate for LIME explainer: (For really long input text, go up to 5000)', value=1000)
# method_list = tuple(label for label, val in METHODS.items())
# method = st.selectbox(
#     'Choose classifier:',
#     method_list,
#     index=4,
#     format_func=format_dropdown_labels,
# )
st.file_uploader("Upload Model",accept_multiple_files = False, help = 'Upload H5 file')
st.text("")
st.file_uploader("Upload Model Class",accept_multiple_files = False, help = 'Upload a Python file that contains just the class of the model that is used')
st.file_uploader('Upload Image that needs to be analyzed')

np.random.seed(123)
def load_image_data_from_url(url):
    '''
    Function to load image data from online
    '''
    # The local path to our target image
    image_path = keras.utils.get_file("shark.jpg", url)

    display(Image(image_path))
    return image_path

image_path = load_image_data_from_url(url = "https://images.unsplash.com/photo-1560275619-4662e36fa65c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1200&q=80")

IMG_SIZE = (299, 299)
def transform_image(image_path, size):
    '''
    Function to transform an image to normalized numpy array
    '''
    img = image.load_img(image_path, target_size=size)
    img = image.img_to_array(img)  # Transforming the image to get the shape as [channel, height, width]
    img = np.expand_dims(img, axis=0)  # Adding dimension to convert array into a batch of size (1,299,299,3)
    img = img / 255.0  # normalizing the image to keep within the range of 0.0 to 1.0

    return img
normalized_img = transform_image(image_path, IMG_SIZE)

from tensorflow.keras.applications.xception import Xception
model = Xception(weights="imagenet")

def get_model_predictions(data):
    model_prediction = model.predict(data)
    print(f"The predicted class is : {decode_predictions(model_prediction, top=1)[0][0][1]}")
    return decode_predictions(model_prediction, top=1)[0][0][1]

#plt.imshow(normalized_img[0])
pred_orig = get_model_predictions(normalized_img)

if st.button("Explain Results"):
    with st.spinner('Calculating...'):
        explainer = lime_image.LimeImageExplainer()
        exp = explainer.explain_instance(normalized_img[0],
                                         model.predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)
        # Display explainer HTML object
        plt.imshow(exp.segments)
        plt.axis('off')
        plt.show()


        def generate_prediction_sample(exp, exp_class, weight=0.1, show_positive=True, hide_background=True):
            '''
            Method to display and highlight super-pixels used by the black-box model to make predictions
            '''
            image, mask = exp.get_image_and_mask(exp_class,
                                                 positive_only=show_positive,
                                                 num_features=6,
                                                 hide_rest=hide_background,
                                                 min_weight=weight
                                                 )
            plt.imshow(mark_boundaries(image, mask))
            plt.axis('off')
            plt.show()


        generate_prediction_sample(exp, exp.top_labels[0], show_positive=True, hide_background=True)

        generate_prediction_sample(exp, exp.top_labels[0], show_positive=True, hide_background=False)