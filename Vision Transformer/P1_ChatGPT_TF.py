import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import time
import os
import logging

# Set up logging
script_name = os.path.basename(__file__).split('.')[0]
log_filename = f'{script_name}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')


# Function to log information
def log(message):
    logging.info(message)
    print(message)


# Record start time
start_time = time.time()
log(f'Starting execution at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

# Load and preprocess MNIST dataset
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)


def preprocess(image, label):
    image = tf.image.resize(image, [32, 32])  # Resize to 32x32 for Vision Transformer
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label


ds_train = ds_train.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)


# Define Vision Transformer (ViT) model
def create_vit_model(input_shape=(32, 32, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Patch creation
    patches = layers.Conv2D(64, kernel_size=4, strides=4, padding='valid')(inputs)
    patches = layers.Reshape((64, 64))(patches)

    # Transformer block
    encoded_patches = layers.LayerNormalization()(patches)
    encoded_patches = layers.MultiHeadAttention(num_heads=4, key_dim=64)(encoded_patches, encoded_patches)
    encoded_patches = layers.Add()([patches, encoded_patches])  # Skip connection

    # Global pooling and classification
    encoded_patches = layers.GlobalAveragePooling1D()(encoded_patches)
    outputs = layers.Dense(num_classes, activation='softmax')(encoded_patches)

    model = tf.keras.Model(inputs, outputs)
    return model


vit_model = create_vit_model()

# Compile the model
vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = vit_model.fit(ds_train, validation_data=ds_test, epochs=5)

# Evaluate the model
test_loss, test_acc = vit_model.evaluate(ds_test)
log(f'Test accuracy: {test_acc}')

# Record end time
end_time = time.time()
log(f'Ending execution at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')

# Calculate total runtime
total_runtime = end_time - start_time
log(f'Total execution time: {total_runtime:.2f} seconds')

# Save the model
vit_model.save(f'{script_name}_vit_model.h5')
log('Model saved successfully.')
