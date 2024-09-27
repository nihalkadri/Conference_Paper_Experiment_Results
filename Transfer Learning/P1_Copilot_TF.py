import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications
import time
import os
import logging

# Set up logging
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file = f"{script_name}.log"
logging.basicConfig(filename=log_file, level=logging.INFO)

# Record start time
start_time = time.time()
logging.info(f"Start time: {time.ctime(start_time)}")

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images[..., tf.newaxis] / 255.0, test_images[..., tf.newaxis] / 255.0

# Resize images to match the input size of the pre-trained model
train_images = tf.image.resize(train_images, (32, 32))
test_images = tf.image.resize(test_images, (32, 32))

# Load the pre-trained model (MobileNetV2) without the top layer
base_model = applications.MobileNetV2(input_shape=(32, 32, 1), include_top=False, weights=None)
base_model.trainable = False

# Add new classification layers on top of the pre-trained model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
logging.info(f"Test accuracy: {test_acc}")

# Record end time
end_time = time.time()
logging.info(f"End time: {time.ctime(end_time)}")

# Calculate and log total runtime
total_runtime = end_time - start_time
logging.info(f"Total runtime: {total_runtime} seconds")

# Display start, end times, and total runtime
print(f"Start time: {time.ctime(start_time)}")
print(f"End time: {time.ctime(end_time)}")
print(f"Total runtime: {total_runtime} seconds")
