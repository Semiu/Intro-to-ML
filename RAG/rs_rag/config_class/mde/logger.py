"""
Module that defines the error and process info logging framework
"""

import logging


def define_logger():
    """
    Defines the logging function
    Returns
        logger class
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    return logger
