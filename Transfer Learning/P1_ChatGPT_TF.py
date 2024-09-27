import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import mnist
import time
import os
import sys

# Start timer
start_time = time.time()

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess dataset: Resize images in batches and cast to float32
def preprocess_images(images, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.batch(batch_size)
    processed_images = []

    for batch in dataset:
        batch = tf.expand_dims(batch, -1)  # Add channel dimension
        batch = tf.image.grayscale_to_rgb(batch)  # Convert grayscale to RGB
        batch = tf.image.resize(batch, [64, 64])  # Resize to 64x64 to reduce memory load
        batch = tf.cast(batch, tf.float32) / 255.0  # Normalize to [0,1]
        processed_images.append(batch)

    return tf.concat(processed_images, axis=0)

# Process images in batches
batch_size = 128  # Adjust the batch size to fit memory constraints
train_images = preprocess_images(train_images, batch_size)
test_images = preprocess_images(test_images, batch_size)

# Create datasets using tf.data API for memory efficiency
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Load pre-trained MobileNetV2 model without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False  # Freeze base layers

# Add custom layers on top of MobileNetV2
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Dropout to prevent overfitting
    layers.Dense(10, activation='softmax')  # 10 output classes for MNIST
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)  # Adjust epochs based on RAM constraints

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# End timer and calculate runtime
end_time = time.time()
total_time = end_time - start_time
print(f"Start time: {time.ctime(start_time)}")
print(f"End time: {time.ctime(end_time)}")
print(f"Total runtime: {total_time:.2f} seconds")

# Save log to a file
log_file = "mnist_transfer_learning.log"
sys.stdout = open(log_file, 'w')
print(f"Test accuracy: {test_acc:.4f}")
print(f"Start time: {time.ctime(start_time)}")
print(f"End time: {time.ctime(end_time)}")
print(f"Total runtime: {total_time:.2f} seconds")
sys.stdout.close()

print(f"Log saved to {log_file}")
