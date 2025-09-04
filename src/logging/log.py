import logging


def get_logger(name=None, level=logging.INFO):
    """
    Returns a reusable logger object with the specified name and level.
    If no name is provided, uses the root logger.
    """
    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(level)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
