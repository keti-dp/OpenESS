# -*- coding: utf-8 -*-
import logging
import ecs_logging
import os
import logs


def get_logger(
    lname=None,
    dirname="./log/",
    # filename="process_log.log",
    filename="process_log.json",
    # format="[%(levelname)s|%(funcName)s] [%(asctime)s] %(message)s",
    format="%(message)s",
):

    if os.path.isdir(dirname) is False:
        os.makedirs(dirname)

    logger = logging.getLogger(lname)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(filename=dirname + filename, mode="a")
    fileHandler.setFormatter(ecs_logging.StdlibFormatter())
    logger.addHandler(fileHandler)

    return logger


def write_log(logger, level, message):
    if level == logging.INFO:
        logger.setLevel(level)
        logger.info(message)
    elif level == logging.DEBUG:
        logger.setLevel(level)
        logger.debug(message)
    elif level == logging.WARNING:
        logger.setLevel(level)
        logger.warning(message)
    elif level == logging.ERROR:
        logger.setLevel(level)
        logger.error(message)
    elif level == logging.CRITICAL:
        logger.setLevel(level)
        logger.critical(message)
    else:
        print("unusable log level: %s" % level)
