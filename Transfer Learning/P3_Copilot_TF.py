import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import time
import logging
import os

# Set up logging
script_name = os.path.splitext(os.path.basename(__file__))[0]
logging.basicConfig(filename=f'{script_name}.log', level=logging.INFO)

# Record start time
start_time = time.time()
logging.info(f'Start time: {time.ctime(start_time)}')

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the base model for transfer learning
base_model = applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    layers.experimental.preprocessing.Resizing(32, 32, interpolation='bilinear', input_shape=(28, 28, 1)),
    layers.Conv2D(3, (3, 3), padding='same', activation='relu'),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
logging.info(f'Test accuracy: {test_acc:.2f}')

# Record end time and total runtime
end_time = time.time()
logging.info(f'End time: {time.ctime(end_time)}')
total_runtime = end_time - start_time
logging.info(f'Total runtime: {total_runtime:.2f} seconds')

print(f'Test accuracy: {test_acc:.2f}')
print(f'Total runtime: {total_runtime:.2f} seconds')
