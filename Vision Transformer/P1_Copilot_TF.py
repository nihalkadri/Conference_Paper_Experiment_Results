
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
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
logging.info('Script started at: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define Vision Transformer model
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(input_shape, num_classes, projection_dim, transformer_layers, num_heads, mlp_head_units):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size=4)(inputs)
    num_patches = (input_shape[0] // 4) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=projection_dim, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = layers.Dense(units=mlp_head_units, activation=tf.nn.gelu)(representation)
    logits = layers.Dense(units=num_classes)(features)
    model = models.Model(inputs=inputs, outputs=logits)
    return model

# Create and compile the model
input_shape = (28, 28, 1)
num_classes = 10
projection_dim = 64
transformer_layers = 8
num_heads = 4
mlp_head_units = 128

model = create_vit_classifier(input_shape, num_classes, projection_dim, transformer_layers, num_heads, mlp_head_units)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
logging.info(f'Test accuracy: {test_acc:.2f}')
logging.info(f'Test loss: {test_loss:.2f}')
print(f'Test accuracy: {test_acc:.2f}')
print(f'Test loss: {test_loss:.2f}')

# Record end time
end_time = time.time()
logging.info('Script ended at: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
logging.info('Total time taken: ' + str(end_time - start_time))
print('Total time taken: ', end_time - start_time)
