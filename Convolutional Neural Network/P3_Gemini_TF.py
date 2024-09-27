import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time
from tensorflow.keras.utils import to_categorical
# Function to log data to a file
def log_data(data, filename):
    with open(filename, "a") as f:
        f.write(f"{data}\n")

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data (resize and normalize)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create CNN model with data augmentation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Start timing
start_time = time.time()

# Train the model with data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=15,  # Increase epochs for better accuracy
                    validation_data=(x_test, y_test))

# End timing
end_time = time.time()

# Calculate total runtime
total_runtime = end_time - start_time

# Log data to a file
log_data(f"Start Time: {start_time}", __file__.replace(".py", ".log"))
log_data(f"End Time: {end_time}", __file__.replace(".py", ".log"))
log_data(f"Total Runtime: {total_runtime} seconds", __file__.replace(".py", ".log"))

# Evaluate the model
score = model.evaluate(x_test, y_test)[1]
print('Test accuracy:', score)

# Log test accuracy
log_data(f"Test Accuracy: {score}", __file__.replace(".py", ".log"))