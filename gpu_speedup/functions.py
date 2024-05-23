import tensorflow as tf

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
