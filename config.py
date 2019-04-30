import logging

logger = logging.getLogger("Word-Vector-Representation-Logger")
logger.setLevel(logging.DEBUG)

stdout_logger = logging.StreamHandler()
stdout_logger.setFormatter(
    logging.Formatter(
        '[%(filename)s:%(lineno)d] - %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
)

logger.addHandler(stdout_logger)
