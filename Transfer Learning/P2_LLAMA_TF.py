import tensorflow as tf
import numpy as np
import time
import logging
import os

# Set up logging
script_name = os.path.basename(__file__).split('.')[0]
log_file = script_name + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO)

# Record start time
start_time = time.time()
logging.info('Script started at: ' + str(start_time))

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_train = tf.image.resize(x_train, (224, 224))  # Resize images to 224x224
x_test = tf.image.resize(x_test, (224, 224))  # Resize images to 224x224
x_train = np.repeat(x_train, 3, axis=-1)  # Repeat grayscale channel to create 3-channel image
x_test = np.repeat(x_test, 3, axis=-1)  # Repeat grayscale channel to create 3-channel image

# Define Transfer Learning model using MobileNetV2
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test),
                    batch_size=128,
                    verbose=2)

# Record end time
end_time = time.time()
logging.info('Script ended at: ' + str(end_time))
logging.info('Total time taken: ' + str(end_time - start_time))

# Print total time taken
print('Total time taken: ', end_time - start_time)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
logging.info('Test accuracy: %.2f' % test_acc)
print('Test accuracy: %.2f' % test_acc)