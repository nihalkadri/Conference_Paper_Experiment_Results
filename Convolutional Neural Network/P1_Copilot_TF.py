import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
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
