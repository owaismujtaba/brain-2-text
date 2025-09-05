import logging
import os

def setup_logger(name, log_file_path, level, format ):
    """
    Returns a reusable logger object with the specified name and level,
    configured to log to both a file and the console.
    If no name is provided, uses the root logger.
    """
    # Configure root logger using basicConfig (applies to all loggers unless overridden)
    logging.basicConfig(
        filename=log_file_path,
        level=level,
        format=format
    )
    #os.makedirs(log_file_path, exist_ok=True)
    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Stream handler for console output
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
