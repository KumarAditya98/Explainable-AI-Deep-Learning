#%%

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from io import BytesIO
from PIL import Image
import os
from xplique.attributions import GradCAMPP, Occlusion, Rise, IntegratedGradients, Lime
from images import plot_attribution
import streamlit as st

os.chdir("/home/ubuntu/attempt2")

st.header("XAI Conv")

if "phase1" not in st.session_state:
    st.session_state["phase1"] = False

if "phase2" not in st.session_state:
    st.session_state["phase2"] = False

if "phase2.5" not in st.session_state:
    st.session_state["phase2.5"] = False

if "phase3" not in st.session_state:
    st.session_state["phase3"] = False

# Phase 1. Let user pick image:
image = st.file_uploader("Chosse an image:")

if image is None:
    st.stop()

image = image.read()
image = Image.open(BytesIO(image))
image.save("downloaded_image.jpg")

st.image(image)

st.session_state["phase1"] = True

if st.session_state["phase1"] == True:
    
    # Phase 2. Let user pick pretrained model
    
    option = st.selectbox(
    "Choose pretrained model:",
    ('Select model', 'InceptionV3', 'ResNet50'))

    if option == 'Select model':
        st.stop()
    
    if option == "InceptionV3":
        model = tf.keras.applications.InceptionV3()
        x = np.expand_dims(tf.keras.preprocessing.image.load_img(os.getcwd() + os.sep + "downloaded_image.jpg", target_size=(299, 299)), 0)
        x = np.array(x, dtype=np.float32) / 255.0

        y = np.expand_dims(tf.keras.utils.to_categorical(277, 1000), 0)
    
    if option == "ResNet50":
        model = tf.keras.applications.ResNet50()
        x = np.expand_dims(tf.keras.preprocessing.image.load_img(os.getcwd() + os.sep + "downloaded_image.jpg", target_size=(224, 224)), 0)
        x = np.array(x, dtype=np.float32) / 255.0

        y = np.expand_dims(tf.keras.utils.to_categorical(277, 1000), 0)

    st.session_state["phase2"] = True

if st.session_state["phase2"] == True:
    
    # Phase 3. Pick XAI model:
    
    option_xai = st.selectbox(
    "Choose pretrained model:",
    ('Select XAI model', 'GradCAM', 'Occlusion Sensitivity', 'Rise', 'Integrated Gradients', 'Lime'))
    
    if option_xai == 'Select XAI model':
        st.stop()
    
    if option_xai == 'GradCAM':
        
        st.subheader("Def.")
        st.markdown("Grad-CAM uses the gradients of any target concept (say logits for “dog” or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.")
        st.markdown("The weights of the last layer k are defined as:")
        st.latex(r'''
                 w_{k} = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial f(x)}{\partial A^{k}_{i,j}}
                 ''')
        st.markdown("Then we aggregate features using the attribution defined as:")
        st.latex(r'''
            \theta = max(0, \sum_{k} w_{k} A^{k})
            ''')
        
        st.link_button("Go to research paper", url = "https://arxiv.org/pdf/1610.02391.pdf")
        st.link_button("Go to source code", url="https://github.com/deel-ai/xplique/blob/master/xplique/attributions/grad_cam_pp.py")
        
        explainer = GradCAMPP(model = model,
                              output_layer=-1,
                              batch_size=16,
                              conv_layer=None)

        explanation = explainer.explain(x, y)
        
        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("GradCAM")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        st.pyplot(fig)
    
    if option_xai == 'Occlusion Sensitivity':
        
        st.subheader("Def.")
        st.markdown("The Occlusion sensitivity method sweep a patch that occludes pixels over the images, and use the variations of the model prediction to deduce critical areas.")
        st.markdown("The Occlusion sensitivity map is defined as:")
        st.latex(r'''
                 \theta_{i} = S_{c}(x) - S_{c}(x_{[x_{i}=\bar{x}]})
                 ''')
        
        st.link_button("Go to research paper", url = "https://arxiv.org/pdf/1711.06104.pdf")
        st.link_button("Go to source code", url="https://github.com/deel-ai/xplique/blob/master/xplique/attributions/occlusion.py")

        with st.form("Occlusion"):
            
            st.write("Enter Patch Size:")
            PATCH_SIZE = st.number_input(" ", step=2)
            st.write("Enter Patch Stride:")
            PATCH_STRIDE = st.number_input(" ", step=1)
            
            submit = st.form_submit_button("Submit")
        
        if submit is False:
            st.stop()
        
        PATCH_SIZE = (PATCH_SIZE, PATCH_STRIDE)
        PATCH_STRIDE = (PATCH_STRIDE, PATCH_STRIDE)
        
        explainer = Occlusion(model,
                              patch_size= PATCH_SIZE, 
                              patch_stride= PATCH_STRIDE,
                              batch_size=16,
                              occlusion_value=0)
        
        explanation = explainer.explain(x, y)
        
        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("Occlusion Sensitivity")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        st.pyplot(fig)
    
    if option_xai == 'Rise':
        
        st.subheader("Def.")
        st.markdown("Randomized Input Sampling for Explanation of Black-box Models")
        st.markdown("The Rise method is a perturbation-based method for computer vision, it generates binary masks and study the behavior of the model on masked images. The pixel influence score is the mean of all the obtained scores when the pixel was not masked.")
        st.link_button("Go to research paper", url="https://arxiv.org/pdf/1806.07421.pdf")
        st.link_button("Go to source code", url="https://github.com/deel-ai/xplique/blob/master/xplique/attributions/rise.py")
        
        st.image("download.png")
        
        with st.form("Rise"):
            
            st.write("Enter the sample number of masks:")
            NB_SAMPLES = st.number_input(" ", step=1000)
            st.write("Enter the grid size:")
            GRID_SIZE = st.number_input(" ", step=1)
            st.write("Enter the probability of pixels to be preserved:")
            PRESERVATION = st.number_input(" ", step=0.1)
            
            submit = st.form_submit_button("Submit")
            
        if submit is False:
            st.stop()
        
        explainer = Rise(model,
                         nb_samples=NB_SAMPLES, 
                         grid_size=GRID_SIZE,
                         preservation_probability=PRESERVATION)

        explanation = explainer.explain(x, y)

        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("Rise")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        st.pyplot(fig)
    
    if option_xai == 'Integrated Gradients':
        
        st.subheader("Def.")
        st.markdown("Integrated Gradients is a visualization technique resulting of a theoretical search for an explanatory method that satisfies two axioms, Sensitivity and Implementation Invariance (Sundararajan et al.)")
        st.link_button("Go to research paper", url="https://arxiv.org/pdf/1703.01365.pdf")
        st.link_button("Go to source code", url="https://github.com/deel-ai/xplique/blob/master/xplique/attributions/integrated_gradients.py")
        
        with st.form("integratedgrad"):
            st.write("Enter the number of steps:")
            STEPS = st.number_input(" ", step=10)
            st.write("Enter the sample number of masks:")
            BASELINE = st.number_input(" ", step=1)
            
            submit = st.form_submit_button("Submit")
        
        if submit == False:
            st.stop()
            
        explainer = IntegratedGradients(model,
                                        output_layer=-1, 
                                        batch_size=16,
                                        steps=STEPS, 
                                        baseline_value=BASELINE)

        explanation = explainer.explain(x, y)

        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("Integrated Gradients")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        st.pyplot(fig)
    
    if option_xai == 'Lime':
        
        st.subheader("Def.")
        st.markdown(
            "The acronym LIME stands for Local Interpretable Model-agnostic Explanations. It is one of the most popular Explainable AI (XAI) methods used for explaining the working of machine learning and deep learning models.")
        st.link_button("Go to research paper", url="https://arxiv.org/abs/1602.04938.pdf")
        st.link_button("Go to source code", url="https://github.com/marcotcr/lime")
        st.markdown("""
    The abbreviation of LIME itself gives an intuition about the core idea behind it. LIME is:
    * Model agnostic, which means that LIME is model-independent. In other words, LIME is able to explain any black-box classifier you can think of.
    * Interpretable, which means that LIME provides you a solution to understand why your model behaves the way it does.
    * Local, which means that LIME tries to find the explanation of your black-box model by approximating the local linear behavior of your model.   
    
    ### How LIME works internally:   
    * Input Data Permutation
    * Predict Class of each artificial Model
    * Calculate weight of each artificial data point
    * Fit a linear classifier to retrieve the most important features   
    
    ### Limitation: The primary limitation of LIME is that currently it supports analysis only for a single image at a time, and not considering the whole dataset.""")

    st.session_state["phase3"] = True
    

    
    
