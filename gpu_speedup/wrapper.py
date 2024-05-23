import tensorflow as tf
import time
from functools import wraps


def device_context(device_name):
    """
    Decorator to set the device context for the computation.

    Args:
        device_name (str): The device to be used ('/cpu:0' or '/device:GPU:0').
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with tf.device(device_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timeit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start timing
        result = func(*args, **kwargs)  # Call the function
        end_time = time.perf_counter()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function {func.__name__!r} executed in {elapsed_time:.4f} seconds.")
        return result, elapsed_time  # Return the function result and the elapsed time

    return wrapper
