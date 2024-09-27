# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import time
import logging
import os
import matplotlib.pyplot as plt

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


# Define Vision Transformer (ViT) model
def vit_model(patch_size=4, num_classes=10, hidden_size=128, transformer_layers=4):
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, (patch_size, patch_size), strides=(patch_size, patch_size))(inputs)
    x = layers.Reshape((7 * 7, 64))(x)
    x = layers.LayerNormalization()(x)

    for _ in range(transformer_layers):
        x = layers.MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
        x = layers.Add()([x, layers.Dropout(0.1)(x)])
        x = layers.LayerNormalization()(x)
        x = layers.Dense(hidden_size, activation='relu')(x)
        x = layers.Add()([x, layers.Dropout(0.1)(x)])
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)


# Compile model
model = vit_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
logging.info(f'Test Accuracy: {test_acc:.4f}')
logging.info(f'Test Loss: {test_loss:.4f}')

# Record end time and calculate runtime
end_time = time.time()
runtime = end_time - start_time
logging.info('Script execution completed')
logging.info(f'Total runtime: {runtime:.2f} seconds')

# Save model
model.save('vit_mnist_model.h5')

