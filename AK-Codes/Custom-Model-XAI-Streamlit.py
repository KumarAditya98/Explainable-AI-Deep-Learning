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
import tensorflow as tf
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
#torch.random.seed(123)
#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
def predict(image_tensor,selected_framework,model):
    if selected_framework == "PyTorch":
        with torch.no_grad():
            output = model(image_tensor)
        return output
    else:
        return model.predict(image_tensor)

def preprocess_image(image, preprocess_fn, Mean_list, Std_list, image_size, PyTorch=True,for_model = True):
    if for_model:
        if PyTorch:
            preprocess = eval(preprocess_fn)
            return preprocess(image).unsqueeze(0)
        else:
            mean = np.array([float(a) for a in Mean_list], dtype=np.float32)
            std = np.array([float(a) for a in Std_list], dtype=np.float32)
            image = tf.image.resize(image, [image_size, image_size])
            img = tf.keras.preprocessing.image.img_to_array(image)
            image = (img - mean) / std
            return np.expand_dims(image,axis=0)
    else:
            if PyTorch:
                preprocess = eval(preprocess_fn)
                return preprocess(image)
            else:
                mean = tf.constant([float(a) for a in Mean_list], dtype=tf.float32)
                std = tf.constant([float(a) for a in Std_list], dtype=tf.float32)
                resized_image = tf.image.resize(image, [image_size, image_size])
                image_tensor = tf.cast(resized_image, dtype=tf.float32)
                image = (image_tensor - mean) / std
                return image

# Build app
def main():
    st.balloons()
    title_text = 'AI Explanability Dashboard'
    st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
    st.subheader("Understand the black-box that is your Image Classification Model")
    st.text("")
    # Upload custom PyTorch model (.pt)
    selected_framework = st.radio("Select the Deep Learning framework:", ["PyTorch", "TensorFlow"])
    selected_model = st.radio("Indicate whether using custom model, pre-trained or pre-trained + custom head:", ["Custom", "Pre-trained","Pre-trained + Custom"])
    if selected_model in ["Pre-trained + Custom","Custom"]:
        model_file = st.file_uploader(
            f"Upload your {selected_framework} model (e.g., .pt for PyTorch, .h5 for TensorFlow)", type=["pt", "h5"],accept_multiple_files=False,
            help="""**Please include the import statement and if necessary to make sure the model exists on our server.
            Add custom heads to "Pre-trained + Custom" as shown below.
            Upload Format**:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '<your_library>']) 
            from tensorflow.keras.applications.xception import Xception
            model = Xception(weights='imagenet')
            (OR)
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 10)""")
        if model_file and selected_framework == "PyTorch":
            with open(os.path.join(os.getcwd(), "model.pt"), "wb") as f:
                f.write(model_file.getbuffer())
        if model_file and selected_framework == "TensorFlow":
            with open(os.path.join(os.getcwd(), "model.h5"), "wb") as f:
                f.write(model_file.getbuffer())

    # Upload custom model architecture (.py)
    if selected_model == "Custom":
        model_architecture_code = st.text_area("Enter your custom model class if used PyTorch to create a custom model")
        st.code(model_architecture_code, language="python")
    if selected_model == "Pre-trained + Custom":
        model_architecture_code = st.text_area("Instantiate pre-trained model with custom head")
        st.code(model_architecture_code, language="python")
    if selected_model == "Pre-trained":
        model_architecture_code = st.text_area("Instantiate pre-trained model with corresponding weights.")
        st.code(model_architecture_code, language="python")

    image_size = int(st.text_input("Enter the image size for your model (e.g., 224)", value="224"))
    Mean_list = (st.text_input("Enter your desired image normalization - Mean", value="0.5, 0.5, 0.5"))
    Std_list = (st.text_input("Enter your desired image normalization - Std", value="0.5, 0.5, 0.5"))
    Mean_list = Mean_list.split(",")
    Std_list = Std_list.split(",")

    if selected_framework == "PyTorch":
        preprocess_fn_code = f"torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\ntorchvision.transforms.Resize(({image_size}, {image_size})),\ntorchvision.transforms.Normalize(\nmean={[float(a) for a in Mean_list]},\nstd={[float(a) for a in Std_list]})])"
    else:
        preprocess_fn_code = f"resized_image = tf.image.resize(image, [{image_size}, {image_size}])\nimage_tensor = tf.cast(resized_image, dtype=tf.float32)\nimage = (image_tensor - {list(float(a) for a in Mean_list)}) / {list(float(a) for a in Std_list)}"

    st.text("Applied pre-processing")
    st.code(preprocess_fn_code, language="python")
    if model_architecture_code is not None and selected_model == "Custom":
        clean_string = re.sub(r'#.*', '', model_architecture_code)
        clean_string = re.sub(r'(\'\'\'(.|\n)*?\'\'\'|"""(.|\n)*?""")', '', clean_string, flags=re.DOTALL)
        pattern = re.compile(r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\):',re.IGNORECASE)
        class_name = pattern.search(clean_string)
        if class_name:
            class_name = class_name.group(1)
    elif model_architecture_code is not None and selected_model in ["Pre-trained + Custom","Pre-trained"]:
        model_name = st.text_input("The name of the model variable you've assigned (e.g model)", value='model')

    image_file = st.file_uploader("Upload the image you want to explain", type=["jpg", "jpeg", "png"])

    if model_architecture_code and image_file:
        if selected_framework == "PyTorch":
            exec(model_architecture_code, globals())

            # Load the PyTorch model
            if selected_model == "Pre-trained + Custom":
                model = globals()[model_name]
                file = torch.load("model.pt", map_location=torch.device(device))
                model.load_state_dict(file)
                st.write(model.fc.weight)
            elif selected_model == "Custom":
                model = globals()[class_name]
                file = torch.load("model.pt", map_location=torch.device(device))
                model.load_state_dict(file)
            model.eval()

        elif selected_framework == "TensorFlow":
            if selected_model == "Pre-trained":
                exec(model_architecture_code, globals())
                model = globals()['model']
                st.write(model.summary())
            else:
                model = tf.keras.models.load_model("model.h5")

        else:
            st.error("Invalid framework selected.")

        # Load and display the image
        image = Image.open(image_file)
        image = image.resize((int(image_size), int(image_size)))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        my_bool = True if selected_framework == "PyTorch" else False
        input_image = preprocess_image(image,preprocess_fn_code,Mean_list,Std_list,image_size,PyTorch=my_bool,for_model = True)

        # Define a function for model prediction
        pred_orig = predict(input_image,selected_framework,model)
        st.write("The Predicted Output from the model is as follows:",pred_orig)

        if st.button("Explain Results"):
            with st.spinner('Calculating...'):
                if selected_framework == "PyTorch":
                    # def get_preprocess_transform():
                    #     normalize = transforms.Normalize(mean=[float(a) for a in Mean_list],
                    #                                      std=[float(a) for a in Std_list])
                    #     transf = transforms.Compose([
                    #         transforms.ToTensor(),
                    #         transforms.Resize((image_size, image_size)),
                    #         normalize
                    #     ])
                    #     return transf
                    #
                    # preprocess_transform = get_preprocess_transform()
                    def batch_predict(images):
                        model.eval()
                        batch = torch.stack(tuple(preprocess_image(i,preprocess_fn_code,my_bool,False) for i in images), dim=0)
                        model.to(device)
                        batch = batch.to(device)
                        logits = model(batch)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        return probs.detach().cpu().numpy()
                    explainer = lime_image.LimeImageExplainer()
                    exp = explainer.explain_instance(np.array(image),
                                                 batch_predict,
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)
                else:
                    explainer = lime_image.LimeImageExplainer()
                    exp = explainer.explain_instance(np.array(input_image[0]),
                                                     model.predict,
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

                #explanation_heatmap(exp, exp.top_labels[0])

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



