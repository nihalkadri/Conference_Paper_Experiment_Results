# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import time
import logging
import os

# Get the script's directory path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging
script_name = os.path.basename(__file__).split('.')[0]
log_file = os.path.join(script_dir, f'{script_name}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

# Record start time
start_time = time.time()
logging.info('Script execution started')

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape and normalize data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define Convolutional Neural Network (CNN) model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                    validation_data=(x_test, y_test), verbose=2)

# Record end time and calculate runtime
end_time = time.time()
runtime = end_time - start_time
logging.info('Script execution completed')
logging.info(f'Total runtime: {runtime:.2f} seconds')

# Save model
model.save('cnn_mnist_model.h5')

# Plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')
logging.info(f'Test accuracy: {test_acc:.2f}')