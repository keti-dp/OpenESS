# -*- coding: utf-8 -*-

"""
Copyright 2021, KETI.

2021-12-06 ver 1.0 logs.py 

ESS데이터 수집 시스템 구축 시 로그 수집을 위한 모듈 코드입니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

import logging
import ecs_logging
import os
import logs


def get_logger(
    lname=None,
    dirname="./log/",
    filename="process_log.json",
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
