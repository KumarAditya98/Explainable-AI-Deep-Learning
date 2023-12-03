import streamlit as st
import streamlit.components.v1 as components
from io import StringIO
#import ast
import re
#from lime_explainer import explainer, tokenizer, METHODS
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision import models
import torchvision
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from lime import lime_image
import shap
import numpy as np
import base64
from io import BytesIO
np.random.seed(123)
#torch.random.seed(123)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def preprocess_image(image, preprocess_fn):
    preprocess = eval(preprocess_fn)  # Evaluate the user-provided code
    return preprocess(image).unsqueeze(0)
# Build app
def main():
    st.balloons()
    title_text = 'AI Explainability Dashboard: Image Classification Models for User uploaded Models and Data'
    st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
    st.text("")
    # Upload custom PyTorch model (.pth)
    selected_framework = st.radio("Select the Deep Learning framework:", ["PyTorch", "TensorFlow"])
    selected_model = st.radio("Indicate whether using custom model or pre-trained:", ["Custom", "Pre-trained"])
    model_file = st.file_uploader(
        f"Upload your {selected_framework} model (e.g., .pt for PyTorch, .h5 for TensorFlow)", type=["pt", "h5"],accept_multiple_files=False)
    if model_file:
        with open(os.path.join(os.getcwd(), f"{model_file.name}.pt"), "wb") as f:
            f.write(model_file.getbuffer())
    #st.write(model_file.read())

    # Upload custom model architecture (.py)
    if selected_model == "Custom":
        model_architecture_file = st.text_area("Enter your custom model class if used PyTorch to create a custom model")
        st.code(model_architecture_file, language="python")
    if selected_model == "Pre-trained":
        model_architecture_file = st.text_area("Enter a function that utilizes pre-trained model used")
        st.code(model_architecture_file, language="python")
    #model_architecture_file = st.file_uploader("Upload your custom model architecture (Python file) if used PyTorch to create a custom model", type=["py"], help = "Upload the PyTorch Class that defines your custom model")
    image_size = st.text_input("Enter your desired image size (e.g., 224)", value="224")
    Mean_list = st.text_input("Enter your desired image normalization - Mean", value="0.485, 0.456, 0.406")
    Std_list = st.text_input("Enter your desired image normalization - Std", value="0.229, 0.224, 0.225")
    preprocess_fn_code = f"torchvision.transforms.Compose([\ntorchvision.transforms.Resize(({image_size}, {image_size})),\ntorchvision.transforms.ToTensor(),\ntorchvision.transforms.Normalize(\nmean=[{Mean_list}],\nstd=[{Std_list}]\n)\n])"
    #preprocess_fn_code = st.text_input("Edit image preprocessing function for your problem:",default_code)
    st.text("Applied pre-processing")
    st.code(preprocess_fn_code, language="python")
    if model_architecture_file is not None and selected_model == "Custom":
        #stringio = StringIO(model_architecture_file.getvalue().decode("utf-8"))
        #string_data = stringio.read()
        clean_string = re.sub(r'#.*', '', model_architecture_file)
        clean_string = re.sub(r'(\'\'\'(.|\n)*?\'\'\'|"""(.|\n)*?""")', '', clean_string, flags=re.DOTALL)
        #st.write(f"The uploaded Model Class is as follows:\n{clean_string}")
        pattern = re.compile(r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\):',re.IGNORECASE)
        class_name = pattern.search(clean_string)
        if class_name:
            class_name = class_name.group(1)
    elif model_architecture_file is not None and selected_model == "Pre-trained":
        # stringio = StringIO(model_architecture_file.getvalue().decode("utf-8"))
        # string_data = stringio.read()
        clean_string = re.sub(r'#.*', '', model_architecture_file)
        clean_string = re.sub(r'(\'\'\'(.|\n)*?\'\'\'|"""(.|\n)*?""")', '', clean_string, flags=re.DOTALL)
        # st.write(f"The uploaded Model Class is as follows:\n{clean_string}")
        pattern = re.compile(r'def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\):', re.IGNORECASE)
        class_name = pattern.search(clean_string)
        if class_name:
            class_name = class_name.group(1)

    image_file = st.file_uploader("Upload the image you want to explain", type=["jpg", "jpeg", "png"])

    if model_file and model_architecture_file and image_file:
        if selected_framework == "PyTorch":
            # Load the custom PyTorch model architecture
            #model_architecture_code = model_architecture_file.read().decode("utf-8")
            exec(model_architecture_file,globals())
            #exec(model_architecture_code)  # Execute the code to define the model architecture

            # Load the PyTorch model
            if selected_model == "Pre-trained":
                model_func = globals()[class_name]
                class ModelWrapper(nn.Module):
                    def __init__(self, model_func):
                        super(ModelWrapper, self).__init__()
                        self.model_func = model_func
                    def forward(self, x):
                        return self.model_func(x)

                model = ModelWrapper(model_func)
            elif selected_model == "Custom":
                model = globals()[class_name]
            file = torch.load(f"{model_file.name}.pt",map_location=torch.device(device))
            model.load_state_dict(file,strict = False)
            model.eval()

        elif selected_framework == "TensorFlow":
            # Load the TensorFlow model
            model = tf.keras.models.load_model(model_file)

        else:
            st.error("Invalid framework selected.")

        # Load and display the image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image User - defined image preprocessing function
        #preprocess_fn = preprocess_fn_code.replace("transforms", "transforms.Compose")
        input_image = preprocess_image(image,preprocess_fn_code)

        # Define a function for model prediction
        def predict(image_tensor):
            if selected_framework == "PyTorch":
                with torch.no_grad():
                    output = model(image_tensor)
                return output.numpy()
            elif selected_framework == "TensorFlow":
                return model.predict(image_tensor)

        pred_orig = predict(input_image)
        st.write("The Predicted Output from the model is as follows:",pred_orig)

        if st.button("Explain Results"):
            with st.spinner('Calculating...'):
                explainer = lime_image.LimeImageExplainer()
                exp = explainer.explain_instance(input_image[0],
                                                 pred_orig,
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)
                # Display explainer HTML object
                st.image(exp.segments)
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
                    st.image(mark_boundaries(image, mask))

                generate_prediction_sample(exp, exp.top_labels[0], show_positive=True, hide_background=True)

                generate_prediction_sample(exp, exp.top_labels[0], show_positive=True, hide_background=False)
                generate_prediction_sample(exp, exp.top_labels[0], show_positive=False, hide_background=False)

                def explanation_heatmap(exp, exp_class):
                    '''
                    Using heat-map to highlight the importance of each super-pixel for the model prediction
                    '''
                    dict_heatmap = dict(exp.local_exp[exp_class])
                    heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
                    st.image(heatmap)

                explanation_heatmap(exp, exp.top_labels[0])

        # Explain with LIME
        # lime_explanation = explain_with_lime(image, predict, class_names=["Class 0", "Class 1"])
        # st.subheader("LIME Explanation")
        # st.image(lime_explanation.image, caption="LIME Explanation", use_column_width=True)
        #
        # # Explain with SHAP
        # shap_values = explain_with_shap(image, predict)
        # st.subheader("SHAP Explanation")
        # shap.image_plot(shap_values, -input_image.numpy(), show=False)
        # st.pyplot()

if __name__ == "__main__":
    main()



