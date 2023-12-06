import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from math import ceil
from time import time

import xplique
from xplique.attributions import Saliency, DeconvNet, GradientInput, GuidedBackprop
from xplique.plots import plot_attributions

import requests

def download_image(image_url, filename):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

# Example Usage
download_image('https://unsplash.com/photos/X2PwqTUpXH8/download?force=true&w=640', 'fox.jpg')

x = np.expand_dims(tf.keras.preprocessing.image.load_img("fox.jpg", target_size=(299, 299)), 0)
x = np.array(x, dtype=np.float32) / 255.0

y = np.expand_dims(tf.keras.utils.to_categorical(277, 1000), 0)

plt.rcParams["figure.figsize"] = [12.5, 5]
plt.imshow(x[0])
plt.axis('off')
plt.show()

# load the model
model = tf.keras.applications.InceptionV3()

# define arbitrary parameters (common for all methods)
parameters = {
    "model": model,
    "output_layer": None,
    "batch_size": 16,
}

# instanciate one explainer for each method
explainers = {
    "Saliency": Saliency(**parameters),
    "DeconvNet": DeconvNet(**parameters),
    "GradientInput": GradientInput(**parameters),
    "GuidedBackprop": GuidedBackprop(**parameters),
}

# iterate on all methods
for method_name, explainer in explainers.items():
    # compute explanation by calling the explainer
    explanation = explainer.explain(x, y)

    # visualize explanation with plot_explanation() function
    print(method_name)
    plot_attributions(explanation, x, img_size=5, cmap='cividis', cols=1, alpha=0.6)
    plt.show()

