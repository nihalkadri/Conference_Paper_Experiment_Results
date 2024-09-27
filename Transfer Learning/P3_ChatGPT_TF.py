import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import time
import os
import sys
import numpy as np

# Record the start time
start_time = time.time()

# Set up logging
script_name = 'mnist_transfer_learning'
log_file = f"{script_name}.log"
sys.stdout = open(log_file, 'w')

# Load and preprocess the MNIST dataset
def load_and_preprocess_mnist():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and add channel dimension for MobileNetV2
    train_images = np.stack([train_images]*3, axis=-1) / 255.0  # Convert to 3-channel RGB
    test_images = np.stack([test_images]*3, axis=-1) / 255.0

    # Resize images to 96x96 to reduce memory usage
    train_images = tf.image.resize(train_images, [96, 96])
    test_images = tf.image.resize(test_images, [96, 96])

    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_mnist()

# Create TensorFlow datasets for efficient memory management
batch_size = 32  # Reduced batch size to minimize memory consumption
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load pre-trained MobileNetV2 and add custom classification layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False  # Freeze base layers to reduce memory and computation

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for MNIST
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# Record the end time
end_time = time.time()

# Print start, end, and total time taken
total_time = end_time - start_time
print(f"Script start time: {time.ctime(start_time)}")
print(f"Script end time: {time.ctime(end_time)}")
print(f"Total time taken: {total_time:.2f} seconds")

# Close the log file and reset stdout
sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Log saved to {log_file}")
