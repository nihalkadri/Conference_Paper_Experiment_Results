import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import time
import os
import sys

# Record the start time
start_time = time.time()

# Set up logging
script_name = os.path.basename(__file__).replace('.py', '')
log_file = f"{script_name}.log"
sys.stdout = open(log_file, 'w')

# Load and preprocess MNIST dataset
def load_and_preprocess_mnist():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Convert to TensorFlow tensors and add channel dimension
    train_images = tf.expand_dims(train_images, -1)  # Add channel dimension
    test_images = tf.expand_dims(test_images, -1)  # Add channel dimension

    # Convert grayscale images to RGB
    train_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(train_images, dtype=tf.float32))
    test_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(test_images, dtype=tf.float32))

    # Resize images to 224x224
    train_images = tf.image.resize(train_images, [224, 224])
    test_images = tf.image.resize(test_images, [224, 224])

    # Normalize images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_mnist()

# Create TensorFlow datasets
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load the pre-trained MobileNetV2 model and add custom classification layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

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

