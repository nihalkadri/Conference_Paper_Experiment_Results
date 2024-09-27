import tensorflow as tf
import numpy as np
import time
import os
import logging

# Configure logging
script_name = os.path.basename(__file__).replace('.py', '')
logging.basicConfig(filename=f'{script_name}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Record start time
start_time = time.time()
logging.info('Execution started.')

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data for CNN (adding a single channel dimension)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
logging.info('Starting training...')
history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
logging.info(f'Test accuracy: {test_acc:.4f}')

# Record end time
end_time = time.time()
total_time = end_time - start_time

# Log runtime details
logging.info('Execution finished.')
logging.info(f'Total runtime: {total_time:.2f} seconds')

# Save the model
model.save(f'{script_name}_model.h5')
logging.info(f'Model saved as {script_name}_model.h5')

# Final logging
logging.info('Training completed.')
print(f"Total runtime: {total_time:.2f} seconds")
