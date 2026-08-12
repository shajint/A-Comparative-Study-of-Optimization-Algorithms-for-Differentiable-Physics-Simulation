# logger.py
import logging


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Ignore JAX warnings
    logging.getLogger("jax").setLevel(logging.ERROR)

    # Create a handler
    handler = logging.StreamHandler()

    # Create a formatter
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add the handler only if none exists, and stop propagation to
    # prevent duplicate log messages.
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger