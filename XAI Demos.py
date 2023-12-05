#%%

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from math import ceil
from time import time
from sklearn import linear_model
from io import BytesIO
from PIL import Image
import os
from xplique.attributions import GradCAM, GradCAMPP, Occlusion, Rise, DeconvNet, Lime, KernelShap
from images import plot_attribution
import streamlit as st

os.chdir("/home/ubuntu/attempt2")

st.header("XAI Demos")

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
    ('Select XAI model', 'GradCAM', 'Occlusion Sensitivity', 'Rise', 'Deconvnet', 'Lime', 'Shap'))
    
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
        
        # Let user pick Occlusion param
        
        explainer = Occlusion(model,
                              patch_size=(12, 12), 
                              patch_stride=(4, 4),
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
        
        # Let user pick Rise param

        st.subheader("Defenition")
        st.markdown("Deconvnet is one of the first attribution method and was proposed in 2013. Its operation is similar to Saliency: it consists in backpropagating the output score with respect to the input, however, at each non-linearity (the ReLUs), only the positive gradient (even of negative activations) are backpropagated.")
        st.markdown("More precisely:")
        st.latex(r'''
                 \frac{\partial f(x)}{\partial f_{l}(x)} =  \frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))} \frac{\partial \text{ReLU}(f_l(x))}{\partial f_{l}(x)}
= \frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))} \odot \mathbb{1}(f_{l}(x))
                 ''')
        st.markdown("With the he indicator function. With Deconvnet, the backpropagation is modified such that :")
        st.latex(r'''
                       \frac{\partial f(x)}{\partial f_{l}(x)} =
\frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))} \odot \mathbb{1}(\frac{\partial f(x)}{\partial \text{ReLU}(f_{l}(x))})
                         ''')
        
        explainer = Rise(model,
                         nb_samples=20000, 
                         grid_size=13,
                         preservation_probability=0.5)

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
    
    if option_xai == 'Deconvnet':

        st.subheader("Defenition")
        st.markdown(
            "The RISE method consist of probing the model with randomly masked versions of the input image and obtaining the corresponding outputs to deduce critical areas.")
        st.markdown("The RISE importance estimator is defined as:")
        st.latex(r'''
                         \phi_i = \mathbb{E}( f(x \odot m) | m_i = 1) 
        \approx \frac{1}{\mathbb{E}(\mathcal{M}) N} \sum_{i=1}^N f(x \odot m_i) m_i
                         ''')
        
        explainer = DeconvNet(model = model,
                              output_layer=None,
                              batch_size=16)

        explanation = explainer.explain(x, y)

        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("Deconvnet")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        st.pyplot(fig)
    
    if option_xai == 'Lime':
        
        explainer = Lime(model = model,
                        batch_size = 16,
                        map_to_interpret_space= None,  
                        nb_samples= 4000,
                        ref_value= None, 
                        interpretable_model= linear_model.Ridge(alpha=2),
                        similarity_kernel= None, 
                        pertub_func= None,  
                        distance_mode= "euclidean",  
                        kernel_width= 45.0,  
                        prob= 0.5)
        
        explanation = explainer.explain(x, y)

        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("Lime")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        
        st.pyplot(fig)
    
    if option_xai == 'Shap':
        st.subheader("Def.")
        st.markdown(
            "The exact computation of SHAP values is challenging. However, by combining insights from current additive feature attribution methods, we can approximate them. We describe two model-agnostic approximation methods, [...] and another that is novel (Kernel SHAP)")
        explainer = KernelShap(model = model,
                               batch_size = 16,
                               map_to_interpret_space= None, 
                               nb_samples= 4000,
                               ref_value= None)
        
        explanation = explainer.explain(x, y)

        fig, axes = plt.subplots(figsize= (8,5))
        plt.sca(axes)

        axes.set_title("Shap")
        axes.axis("off")
        plot_attribution(explanation=explanation,
                         image= x,
                         ax = axes, 
                         cmap='cividis',
                         alpha=0.6)
        
        st.pyplot(fig) 

    st.session_state["phase3"] = True
    

    
    
    




##%%

# NEXT PHASE FOR GRADCAM

# # Method = GradCAM
# Method = GradCAMPP

# batch_size = 16
# # None will select "conv2d_93" (cf architecture)
# # Most other values will make warnings or errors
# conv_layers = [None, "conv2d_94", "conv2d_95", "conv2d_96", "conv2d_97", "conv2d_98"]

# for conv_layer in conv_layers:
#     t = time()
#     explainer = Method(model,
#                        batch_size=batch_size,
#                        conv_layer=conv_layer)

#     explanation = explainer.explain(x, y)
#     print(f"conv_layer: {conv_layer} -> {round(time()-t, 4)}s")

#     plot_attributions(explanation, x, img_size=5, cmap='cividis', alpha=0.6)
#     plt.show()


# NEXT PHASE FOR OCCLUSION

# batch_size = 16
# patch_sizes = [72, 36, 18, 9]
# # patch_stride set to a third of patch size, see next section for justifications
# occlusion_value = 0

# for patch_size in patch_sizes:
#     t = time()
#     explainer = Occlusion(model, patch_size=patch_size, patch_stride=patch_size//3,
#                           batch_size=batch_size, occlusion_value=occlusion_value)
#     explanation = explainer.explain(x, y)
#     print(f"patch_size: {patch_size} -> {round(time()-t, 4)}s")

#     plot_attributions(explanation, x, img_size=5, cmap='cividis', alpha=0.6)
#     plt.show()

# # %%
