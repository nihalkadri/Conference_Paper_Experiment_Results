import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
import time
import os

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Load the pre-trained VGG16 model (excluding the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the pre-trained model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Start timing
start_time = time.time()

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# End timing
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

# Create a log file with the same name as the script
filename = os.path.basename(__file__)
log_filename = filename.split('.')[0] + '.log'

# Write the results to the log file
with open(log_filename, 'w') as f:
    f.write(f"Total time taken: {total_time:.2f} seconds\n")
    f.write(f"Test accuracy: {test_acc:.2f}\n")

# Print the results
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Test accuracy: {test_acc:.2f}")