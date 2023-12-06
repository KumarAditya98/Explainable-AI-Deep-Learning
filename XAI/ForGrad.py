import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from math import ceil
from time import time
import keras
import xplique
from xplique.attributions import Rise
from xplique.plots import plot_attributions
import cv2
import requests

def download_image(image_url, filename):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

# Example Usage
download_image('https://unsplash.com/photos/X2PwqTUpXH8/download?force=true&w=640', 'fox1.jpg')
download_image('https://unsplash.com/photos/tIfrzHxhPYQ/download?force=true&w=640', 'fox2.jpg')
download_image('https://unsplash.com/photos/LVnJlyfa7Zk/download?force=true&w=640', 'sea_turtle.jpg')
download_image('https://unsplash.com/photos/sSEEbAzB6fU/download?force=true&w=640', 'lynx.jpg')
download_image('https://unsplash.com/photos/41dAczoRYJY/download?force=true&w=640', 'cat.jpg')
download_image('https://unsplash.com/photos/axqTLZ12Jss/download?force=true&w=640', 'otter.jpg')

img_list = [
    ('fox1.jpg', 277),
    ('fox2.jpg', 277),
    ('sea_turtle.jpg', 33),
    ('lynx.jpg', 287),
    ('cat.jpg', 281),
    ('otter.jpg', 360)
]

def central_crop_and_resize(img, size=224):
  """
  Given a numpy array, extracts the largest possible square and resizes it to
  the requested size
  """
  h, w, _ = img.shape

  min_side = min(h, w)
  max_side_center = max(h, w) // 2.0

  min_cut = int(max_side_center-min_side//2)
  max_cut = int(max_side_center+min_side//2)

  img = img[:, min_cut:max_cut] if w > h else img[min_cut:max_cut]
  img = tf.image.resize(img, (size, size))

  return img

X = []
Y = []

for img_name, label in img_list:
    img = cv2.imread(img_name)[..., ::-1] # when cv2 load an image, the channels are inversed
    img = central_crop_and_resize(img)
    label = tf.keras.utils.to_categorical(label, 1000)

    X.append(img)
    Y.append(label)

X = np.array(X, dtype=np.float32)
Y = np.array(Y)

plt.rcParams["figure.figsize"] = [15, 6]
for img_id, img in enumerate(X):
  plt.subplot(1, len(X), img_id+1)
  plt.imshow(img/255.0)
  plt.axis('off')

import tensorflow.keras.applications as app

model, preprocessing = app.MobileNetV2(classifier_activation="linear"), app.mobilenet_v2.preprocess_input
X_preprocessed = preprocessing(np.array(X, copy=True))

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod)
from xplique.commons import forgrad

batch_size = 64
explainers = [
             Saliency(model),
             GradientInput(model),
             GuidedBackprop(model),
             IntegratedGradients(model, steps=80, batch_size=batch_size),
             SmoothGrad(model, nb_samples=80, batch_size=batch_size),
             SquareGrad(model, nb_samples=80, batch_size=batch_size),
             VarGrad(model, nb_samples=80, batch_size=batch_size),
             Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
]

for explainer in explainers:

  explanations = explainer(X_preprocessed, Y)

  print(f"Method: {explainer.__class__.__name__}")
  plot_attributions(explanations, X, img_size=2., cmap='jet', alpha=0.4,
                    cols=len(X), absolute_value=True, clip_percentile=0.5)
  plt.show()

  filtered_explanations = forgrad(explanations, sigma=15)
  print('With ForGRAD')
  plot_attributions(filtered_explanations, X, img_size=2., cmap='jet', alpha=0.4,
                    cols=len(X), absolute_value=True, clip_percentile=0.5)
  plt.show()

  print("\n")

from xplique.plots import plot_attribution

def show(img, **kwargs):
  img = np.array(img)
  img -= img.min(); img / img.max()
  plt.imshow(img, **kwargs)


explainer = SmoothGrad(model)
explanations = explainer(X_preprocessed, Y)

filtered_explanations = []

sigmas=[224, 200, 125, 100, 50, 30, 20, 15, 7, 4]

for sigma in sigmas:
  filtered_explanation = forgrad(explanations, sigma=sigma)
  filtered_explanations.append(filtered_explanation)

for x_i, x in enumerate(X):
  for sigma_i, sigma in enumerate(sigmas):

    plt.subplot(1, len(sigmas), sigma_i+1)
    plot_attribution(filtered_explanations[sigma_i][x_i], X[x_i], cmap='jet', alpha=0.4,
                     absolute_value=True, clip_percentile=0.5)
    plt.title(f'sigma={sigma}')

  plt.show()

print("\n")