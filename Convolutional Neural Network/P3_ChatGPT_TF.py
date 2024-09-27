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

# Load and preprocess the MNIST dataset (for sign classification)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add a channel dimension for the CNN (since MNIST images are grayscale)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode the labels (10 sign classes)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define a CNN model optimized for high accuracy
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout to reduce overfitting
        tf.keras.layers.Dense(10, activation='softmax')  # 10 sign classes output
    ])
    return model

# Instantiate the model
model = create_model()

# Compile the model with optimizations
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Optimized learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for high accuracy and preventing overfitting
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'{script_name}_best_model.h5', save_best_only=True)
]

# Log model architecture
logging.info('Model architecture created.')

# Train the model, use validation split to track performance
logging.info('Training started...')
history = model.fit(x_train, y_train, epochs=20, batch_size=64,
                    validation_split=0.2, callbacks=callbacks)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
logging.info(f'Test accuracy: {test_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# Record end time
end_time = time.time()
total_time = end_time - start_time

# Log total runtime and save final model
logging.info('Execution finished.')
logging.info(f'Total runtime: {total_time:.2f} seconds')
logging.info(f'Model saved as {script_name}_final_model.h5')

# Save the final model
model.save(f'{script_name}_final_model.h5')

# Display total runtime
print(f"Total runtime: {total_time:.2f} seconds")
