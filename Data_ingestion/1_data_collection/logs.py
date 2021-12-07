# -*- coding: utf-8 -*-

"""
logs.py : 로그수집을 위한 모듈

Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

        ---------------------------------------------------------------------------
        Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0 ver
최신 안정화 버전 : 1.0 ver

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
