import tensorflow as tf
import numpy as np
from typing import Tuple, Any

from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit

from gpu_speedup.wrapper import timeit_decorator


def matrix_elem_sum() -> tf.Tensor:
    """
    Perform a convolution operation on a random image tensor.

    Returns:
        The sum of the elements in the output tensor after the convolution.
    """
    random_image = tf.random.normal((100, 100, 100, 3))
    model = tf.keras.layers.Conv2D(32, 7)(random_image)
    return tf.math.reduce_sum(model)


def matrix_multiplication() -> tf.Tensor:
    """
    Perform a series of tensor operations.

    Returns:
        The sum of the result of the tensor operation.
    """
    matrix1 = tf.random.normal((1000, 1000))
    matrix2 = tf.random.normal((1000, 1000))
    product = tf.linalg.matmul(matrix1, matrix2)
    return tf.math.reduce_sum(product)


def double_well(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(y) + (x ** 2 + y ** 2) / 10


@timeit_decorator
def find_min_np(N: int = 10000, low_val: float = -10, high_val: float = 10) -> tuple[
    ndarray[Any, dtype[floating[_64Bit] | float_]], ndarray[Any, dtype[floating[_64Bit] | float_]], ndarray[Any, Any]]:
    x_array = np.random.uniform(low_val, high_val, N)
    y_array = np.random.uniform(low_val, high_val, N)
    z_array = double_well(x_array, y_array)
    min_index = np.argmin(z_array)
    min_x = x_array[min_index]
    min_y = y_array[min_index]
    min_z = z_array[min_index]
    return min_x, min_y, min_z


@timeit_decorator
def find_min_tf(N: int = 10000, low_val: float = -10, high_val: float = 10) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x_tensor = tf.random.uniform(shape=(N,), minval=low_val, maxval=high_val)
    y_tensor = tf.random.uniform(shape=(N,), minval=low_val, maxval=high_val)
    z_tensor = double_well(x_tensor, y_tensor)  # Ensure double_well is compatible with TensorFlow operations
    min_index = tf.argmin(z_tensor)
    min_z = tf.reduce_min(z_tensor)
    min_x = tf.gather(x_tensor, min_index)
    min_y = tf.gather(y_tensor, min_index)
    return min_x, min_y, min_z
