import tensorflow as tf
import numpy as np

from gpu_speedup.wrapper import timeit_decorator


def matrix_elem_sum():
    """
    Perform a convolution operation on a random image tensor.

    Returns:
        The sum of the elements in the output tensor after the convolution.
    """

    # Create a random image tensor of shape (100, 100, 100, 3)
    random_image = tf.random.normal((100, 100, 100, 3))
    # Apply a 2D convolution operation with 32 filters and a kernel size of 7
    model = tf.keras.layers.Conv2D(32, 7)(random_image)
    # Return the sum of all elements in the output tensor
    return tf.math.reduce_sum(model)


def matrix_multiplication():
    """
    Perform a series of tensor operations.

    Returns:
        The sum of the result of the tensor operation.
    """
    # Example operation: matrix multiplication
    matrix1 = tf.random.normal((1000, 1000))
    matrix2 = tf.random.normal((1000, 1000))
    product = tf.linalg.matmul(matrix1, matrix2)
    return tf.math.reduce_sum(product)


def double_well(x, y):
    return np.sin(x) * np.cos(y) + (x ** 2 + y ** 2) / 10


@timeit_decorator
def find_min_np():
    N = 10000

    # Generate random samples within the domain
    x_array = np.random.uniform(-10, 10, N)
    y_array = np.random.uniform(-10, 10, N)

    # Evaluate the function at each sample point
    z_array = double_well(x_array, y_array)

    # Find the minimum value and the corresponding (x, y)
    min_index = np.argmin(z_array)
    min_x = x_array[min_index]
    min_y = y_array[min_index]
    min_z = z_array[min_index]

    return min_x, min_y, min_z


@timeit_decorator
def find_min_tf():
    # Define the number of iterations/samples
    N = 100000

    # Generate random samples within the domain using TensorFlow
    x_tensor = tf.random.uniform(shape=(N,), minval=-10, maxval=10)
    y_tensor = tf.random.uniform(shape=(N,), minval=-10, maxval=10)

    # Evaluate the function at each sample point
    z_tensor = double_well(x_tensor, y_tensor)

    # Find the minimum value and the corresponding (x, y)
    min_index = tf.argmin(z_tensor)
    min_z = tf.reduce_min(z_tensor)
    min_x = tf.gather(x_tensor, min_index)
    min_y = tf.gather(y_tensor, min_index)

    return min_x, min_y, min_z
