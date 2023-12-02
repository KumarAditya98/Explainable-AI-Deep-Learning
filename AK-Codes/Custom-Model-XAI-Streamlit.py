import streamlit as st
import streamlit.components.v1 as components
from io import StringIO
import ast
import re
#from lime_explainer import explainer, tokenizer, METHODS
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
torch.random.seed(123)
device = 'gpu' if torch.cuda.is_available() else 'cpu'
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
    model_file = st.file_uploader(
        f"Upload your {selected_framework} model (e.g., .pth for PyTorch, .h5 for TensorFlow)", type=["pth", "h5"],accept_multiple_files=False)

    # Upload custom model architecture (.py)
    model_architecture_file = st.text_area("Input your custom model class if used PyTorch to create a custom model")
    #model_architecture_file = st.file_uploader("Upload your custom model architecture (Python file) if used PyTorch to create a custom model", type=["py"], help = "Upload the PyTorch Class that defines your custom model")
    preprocess_fn_code = st.text_input("Edit image preprocessing function for your problem:",f"transforms.Compose([transforms.Resize((224, 224)),\ntransforms.ToTensor(),\ntransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])")
    if model_architecture_file is not None:
        #stringio = StringIO(model_architecture_file.getvalue().decode("utf-8"))
        #string_data = stringio.read()
        clean_string = re.sub(r'#.*', '', model_architecture_file)
        clean_string = re.sub(r'(\'\'\'(.|\n)*?\'\'\'|"""(.|\n)*?""")', '', clean_string, flags=re.DOTALL)
        #st.write(f"The uploaded Model Class is as follows:\n{clean_string}")
        pattern = re.compile(r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*:',re.IGNORECASE)
        match = pattern.search(clean_string)
        class_name = match.group(1)

    image_file = st.file_uploader("Upload the image you want to explain", type=["jpg", "jpeg", "png"])

    if model_file and model_architecture_file and image_file:
        if selected_framework == "PyTorch":
            # Load the custom PyTorch model architecture
            #model_architecture_code = model_architecture_file.read().decode("utf-8")
            exec(model_architecture_file)
            #exec(model_architecture_code)  # Execute the code to define the model architecture

            # Load the PyTorch model
            model = globals()[class_name]
            model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
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
        preprocess_fn = preprocess_fn_code.replace("transforms", "transforms.Compose")
        input_image = preprocess_image(image)

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



