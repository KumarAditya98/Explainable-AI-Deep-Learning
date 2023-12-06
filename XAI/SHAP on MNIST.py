# This default renderer is used for sphinx docs only. Please delete this cell in IPython.
import plotly.io as pio
pio.renderers.default = "png"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from omnixai.data.image import Image
from omnixai.explainers.vision import ShapImage

# Load the MNIST dataset
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Use `Image` objects to represent the training and test datasets
train_imgs, train_labels = Image(x_train.astype('float32'), batched=True), y_train
test_imgs, test_labels = Image(x_test.astype('float32'), batched=True), y_test

preprocess_func = lambda x: np.expand_dims(x.to_numpy() / 255, axis=-1)

batch_size = 128
num_classes = 10
epochs = 10

x_train = preprocess_func(train_imgs)
x_test = preprocess_func(test_imgs)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(num_classes))

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

explainer = ShapImage(
    model=model,
    preprocess_function=preprocess_func
)

explanations = explainer.explain(test_imgs[0:5])
a = explanations._plotly_figure(index=4)
a.write_image("SHAPplot.png")
