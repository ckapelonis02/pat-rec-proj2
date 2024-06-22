from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plots import plot_some_data, plot_some_predictions

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model.
train_images = train_images / 255.0
test_images = test_images / 255.0

plot_some_data(train_images, train_labels, class_names)

# Build the model of dense neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
opt_list = ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax', 'ftrl']
for opt in opt_list:
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    #Train the model
    # Training the neural network model requires the following steps:

    #   1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
    #   2. The model learns to associate images and labels.
    #   3. You ask the model to make predictions about a test set—in this example, the test_images array.
    #   4. Verify that the predictions match the labels from the test_labels array.

    model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

    # Evaluate train accuracy
    train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
    print('\nTrain accuracy:', train_acc, "for optimizer", opt)

    # Evaluate test accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc, "for optimizer", opt)

    # Make predictions
    # With the model trained, you can use it to make predictions about some images.
    # The model's linear outputs, logits.
    # Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    plot_some_predictions(test_images, test_labels, predictions, class_names, num_rows=5, num_cols=3)
