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