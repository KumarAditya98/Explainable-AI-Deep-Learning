from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator,DirectoryIterator
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from keras.applications import ResNet50
import os
from PIL import Image
from tensorflow.keras.utils import plot_model
import warnings

warnings.filterwarnings("ignore")

dir_ = os.path.join('dataset/animals/animals/')

train_datagen = ImageDataGenerator(
    fill_mode='nearest',
    validation_split=0.1
)

# Train, validation, and test splits
train_generator = train_datagen.flow_from_directory(
    dir_,
    target_size=(108, 108),
    color_mode='rgb',
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dir_,
    target_size=(108, 108),
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

test_generator = train_datagen.flow_from_directory(
    dir_,
    target_size=(108, 108),
    color_mode='rgb',
    class_mode='categorical',
    subset='validation'
)

model = tf.keras.models.Sequential([
    ResNet50(input_shape=(108, 108, 3), include_top=False),])
for layer in model.layers:
    layer.trainable = False

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    verbose=1
)
model.save('model_custom.h5')

# Making a prediction
from skimage import io
from tensorflow.keras.preprocessing import image
from tensorflow import keras

url = 'https://raw.githubusercontent.com/marcellusruben/All_things_medium/main/Lime/panda_00024.jpg'
def load_image_data_from_url(url):
    image_path = keras.utils.get_file("panda.jpg", url)
    return image_path

image_path = load_image_data_from_url(url)
def read_and_transform_img(image_path):
    img = image.load_img(image_path, target_size = (108,108))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img
images = read_and_transform_img(image_path)
preds = model.predict(images)
prediction = np.argmax(preds)
pct = np.max(preds)

if prediction == 0:
    print('It\'s a cat!')
elif prediction == 1:
    print('It\'s a dog!')
else:
    print('It\'s a panda!')
print(pct)

from tensorflow.keras.applications import inception_v3 as inc_net
inet_model = inc_net.InceptionV3()
inet_model.save('model_inception.h5')

import skimage
from tensorflow.keras.applications.imagenet_utils import decode_predictions
def transform_img_fn_ori(url):
    img = skimage.io.imread(url)
    img = skimage.transform.resize(img, (299, 299))
    img = (img - 0.5) * 2
    img = np.expand_dims(img, axis=0)
    preds = inet_model.predict(img)
    for i in decode_predictions(preds)[0]:
        print(i)
    return img

images_inc_im = transform_img_fn_ori(url)
