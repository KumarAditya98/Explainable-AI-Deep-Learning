import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from math import ceil
from time import time

import xplique
from xplique.attributions import Rise
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

# instanciate explainer with arbitrary parameters
explainer = Rise(model,
                 nb_samples=20000, grid_size=13,
                 preservation_probability=0.5)

# compute explanation by calling the explainer
explanation = explainer.explain(x, y)

# visualize explanation with plot_explanation() function
plot_attributions(explanation, x, img_size=5, cmap='cividis', cols=1, alpha=0.7)

batch_size = 16
nb_samples_list = [100, 500, 2000, 5000, 20000]
grid_size = 13
preservation_probability = 0.5

for nb_samples in nb_samples_list:
    t = time()
    explainer = Rise(model, nb_samples=nb_samples, grid_size=grid_size,
                     preservation_probability=preservation_probability)
    explanation = explainer.explain(x, y)
    print(f"nb_samples: {nb_samples} -> {round(time()-t, 4)}s")

    plot_attributions(explanation, x, img_size=5, cmap='cividis', alpha=0.7)
    plt.show()

batch_size = 16
nb_samples = 10000
grid_sizes = [2, 6, 7, 13, 18, 23, 40]  # images are 299x299 and 299 = 13*23
preservation_probability = 0.5

for grid_size in grid_sizes:
    t = time()
    explainer = Rise(model, nb_samples=nb_samples, grid_size=grid_size,
                     preservation_probability=preservation_probability)
    explanation = explainer.explain(x, y)
    print(f"grid_size: {grid_size} -> {round(time()-t, 4)}s")

    plot_attributions(explanation, x, img_size=5, cmap='cividis', alpha=0.7)
    plt.show()

batch_size = 16
nb_samples = 4000
grid_size = 13
preservation_probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]

for preservation_probability in preservation_probabilities:
    t = time()
    explainer = Rise(model, nb_samples=nb_samples, grid_size=grid_size,
                     preservation_probability=preservation_probability)
    explanation = explainer.explain(x, y)
    print(f"preservation_probability: {preservation_probability} -> {round(time()-t, 4)}s")

    plot_attributions(explanation, x, img_size=5, cmap='cividis', alpha=0.7)
    plt.show()