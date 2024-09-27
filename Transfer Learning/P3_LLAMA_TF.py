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

# Reshape and normalize data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)
])

# Define base model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False
)

# Freeze base model layers
base_model.trainable = False

# Define classification model
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Resizing(224, 224)(inputs)
x = tf.keras.layers.Concatenate()([x, x, x])  # Repeat grayscale channel
x = data_augmentation(x)
x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

# Define complete model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model with mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model with batch size 8 and gradient checkpointing
batch_size = 8
checkpoint_dir = 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_freq=1000  # Save every 1000 batches
)

model.fit(x_train, y_train, epochs=5,
          validation_data=(x_test, y_test),
          verbose=2, batch_size=batch_size,
          callbacks=[checkpoint_callback])

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