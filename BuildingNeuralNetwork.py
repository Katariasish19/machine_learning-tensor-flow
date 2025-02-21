### Building a Neural Network ###

import tensorflow as tf
from tensorflow import keras
from keras import  layers, datasets, models

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Normalise the images, Pre-processing the data!
train_images, test_images= train_images/255.0 , test_images/255.0

# Build the model
model = models.Sequential([               # Sequential is Stack of Layers in Neural Network so that it will allow every layer as 1 element
    layers.Flatten(input_shape=(28,28)),  # 28*28 = 784 values, Converting 2D to 1D array
    layers.Dense(128, activation='relu'), # Has 128 neurons , relu = Rectified Linear Unit
    layers.Dense(10)                      #  Has 10 neurons i.e 0-9 , we want to classify the image into 10 different classes
])

'''
    Before training we should compile the model by specifying how it will learn.
    We define the optimizer, loss function, and metrics while compiling the model.
'''

# Compile the model
model.compile(optimizer='adam', # Optimizer helps adjust weights to reduce errors.
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Loss function measures how far
              metrics = ['accuracy']) # Metrics are used to evaluate the model's performance. To track accuracy


# Train the model
model.fit(train_images, train_labels, epochs=5)  # Train the model for 5 epochs (passes over the data).


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)


              



