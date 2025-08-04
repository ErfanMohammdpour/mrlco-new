import os
import tensorflow as tf

# Test basic GPU setup
print("TensorFlow version:", tf.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Found {len(gpus)} GPU(s)")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu}")
        except RuntimeError as e:
            print(f"Failed to enable memory growth for {gpu}: {e}")

# Test simple operation
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.compat.v1.Session(config=config) as sess:
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    result = sess.run(c)
    print("Matmul result:", result)
    print("Test completed successfully!")