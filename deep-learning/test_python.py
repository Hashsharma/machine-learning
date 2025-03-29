import tensorflow as tf

# Check if TensorFlow is using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus}")
else:
    print("Using CPU.")