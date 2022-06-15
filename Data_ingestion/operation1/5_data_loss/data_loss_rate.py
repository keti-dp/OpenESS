# -*- coding:utf-8 -*-

"""
data_loss_rate.py : 데이터 손실률 체크 모듈

Copyright(C) 2022, 윤태일 / KETI / taeil777@keti.re.kr

        ---------------------------------------------------------------------------
        The 3-Clause BSD 라이선스
        SPDX 단축 식별자: BSD-3-Clause
        참고: 본 라이선스는 "신규 BSD 라이선스"또는 "수정 BSD 라이선스"라고 한다. 2-clause BSD
        라이선스 역시 참조한다.

        저작권 2022 윤태일

        다음 조건이 충족되는 경우, 수정하거나 또는 수정하지 않고 소스 및 바이너리 형식으로 재 배포 및
        사용할 수 있다.

        1. 소스 코드의 재 배포에는 위 저작권 고지, 본 조건 목록 및 아래 면책 조항을 공지해야 한다.
        2. 바이너리 형식으로 재 배포할 경우, 배포 시 제공하는 설명서 및/또는 기타 자료에 위 저작권 고지,
        본 조건 목록 및 다음 면책사항을 공지해야 한다.
        3. 사전 서면 허가 없이, 저작권자의 이름 또는 기여자의 이름을 사용하여, 본 소프트웨어에서 파생된
        제품을 보증 또는 홍보할 수 없다.

        면책 조항:
        본 소프트웨어는 저작권 소유자 및 기여자에 의해 "있는 그대로" 제공되며, 상품성, 특정 목적에의
        적합성에 대한 묵시적 보증 포함 (이에 한정되지는 않음) 명시적 또는 묵시적 보증을 배제한다. 어떠한
        경우에도 저작권 소유자 또는 기여자는 계약, 엄격 책임 또는 (과실 및 기타 사유 포함) 불법 행위 등
        사유 및 책임 이론에 관계없이, (대체 제품 또는 서비스 조달; 사용, 데이터 또는 이익 상실; 사업 중단
        포함; 이에 한정되지는 않음) 본 소프트웨어 사용 관련 직접, 간접, 파생적, 특별, 징벌적 또는 결과적
        손해에 대해 책임을 지지 않는다. 이러한 손해 가능성을 사전에 알고 있은 경우도 마찬가지이다.
        ---------------------------------------------------------------------------


        ---------------------------------------------------------------------------
        The 3-Clause BSD License
        SPDX short identifier: BSD-3-Clause
        Note: This license has also been called the "New BSD License" or "Modified BSD License". See also
        the 2-clause BSD License.

        Copyright 2022 Tae Il Yun

        Redistribution and use in source and binary forms, with or without modification, are permitted
        provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
        following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
        and the following disclaimer in the documentation and/or other materials provided with the
        distribution.

        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
        or promote products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
        TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
        THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

수집된 데이터의 데이터 손실률 체크 코드입니다.

1. 지정된 기간동안의 총 데이터 손실률
2. 하루의 데이터 손실률
3. 지정된 기간동안 하루하루의 데이터 손실률


각 GCP, local 에서 저장되는 운영사이트의 데이터 손실률을 계산합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

from tracemalloc import start
import timescale_input_test
from pytz import timezone
from datetime import date, datetime, timedelta
from datetime import datetime
import logs
import pandas as pd
import sys
import pandas.io.sql as psql

# 1. db 객체를 두 개 만듦
# 2. 1번 db에서 지정된 시간 사이에 count를 가져옴 > 86400개 (1일당 86400개를 기준으로 손실률 계산)
# bank, rack, pcs, etc
#

# 1)매일같이 크론으로 실행할 거
# 2)날짜 입력하면 거기서부터 알아서 할거


def convert_bytes(bytes_number):
    """바이트숫자를 입력하면 단위에 맞게 변환해준다.
    Args:
        bytes_number (int): 변환할 바이트 값
    Returns:
        str: 변환된 값에 단위를 붙여서 반환
    """
    tags = ["Byte", "Kilobyte", "Megabyte", "Gigabyte", "Terabyte"]

    i = 0
    double_bytes = bytes_number

    while i < len(tags) and bytes_number >= 1024:
        double_bytes = bytes_number / 1024.0
        i = i + 1
        bytes_number = bytes_number / 1024

    return str(round(double_bytes, 2)) + " " + tags[i]


def calc_loss_rate(database, operating_site, start_time, end_time, table_name):

    if database == timescale_local:
        DB_region = "local_server1"
    elif database == timescale_GCP:
        DB_region = "GCP_server1"

    bankdivide_var = 1
    if operating_site == "ESS_Operating_Site1":
        if table_name == "rack":
            bankdivide_var = 8
    elif operating_site == "ESS_Operating_Site2":
        if table_name == "rack":
            bankdivide_var = 17
        elif table_name == "bank":
            bankdivide_var = 2

    query_result = database.query_data(
        """SET TIME ZONE 'Asia/Seoul';
        select count(*) from {table_name} where "TIMESTAMP" between '{start_time}' and '{end_time}';""".format(
            table_name=table_name, start_time=start_time, end_time=end_time
        )
    )

    hypertable_size = database.query_data(
        """SET TIME ZONE 'Asia/Seoul';
        SELECT hypertable_size('{table_name}');""".format(
            table_name=table_name
        )
    )[0][0]

    hypertable_size = convert_bytes(hypertable_size)

    table_size_log_message = (
        """{DB_region} / {oper} / {table} data size : {hypertable_size}""".format(
            DB_region=DB_region,
            oper=operating_site,
            table=table_name,
            hypertable_size=hypertable_size,
        )
    )

    dataloss_logger.info(table_size_log_message)

    data_count = query_result[0][0] / bankdivide_var

    if data_count > 86400:
        data_count = 86400

    date_diff = end_time - start_time

    total_daily_data_count = 86400 * date_diff.days

    loss_rate = round(
        ((total_daily_data_count - data_count) / total_daily_data_count * 100), 2
    )

    loss_rate_log_message = """{start_time} / {DB_region} / {oper} / {table} data loss rate : {loss_rate} %""".format(
        start_time=start_time,
        DB_region=DB_region,
        oper=operating_site,
        table=table_name,
        loss_rate=loss_rate,
    )

    # 5퍼센트보다 손실률이 크면 크리티컬
    # 2퍼센트보다 크면 워닝
    # 나머지 인포
    if loss_rate > 5:
        dataloss_logger.critical(loss_rate_log_message)
    elif loss_rate > 2:
        dataloss_logger.warning(loss_rate_log_message)
    else:

        loss_rate_log_message = """{start_time} / {DB_region} / {oper} / {table} data loss rate : {loss_rate} %""".format(
            start_time=start_time,
            DB_region=DB_region,
            oper=operating_site,
            table=table_name,
            loss_rate=loss_rate,
        )
        dataloss_logger.info(loss_rate_log_message)

    print(start_time, "~", end_time)
    print(date_diff.days, "일간", table_name, "수집률 및 손실률")
    print("collection rate :", 100 - loss_rate, "%")
    print("loss rate :", loss_rate, "%")


if __name__ == "__main__":

    operating_site1 = "ESS_Operating_Site1"
    operating_site2 = "ESS_Operating_Site2"
    operating_site_list = []
    operating_site_list.append(operating_site1)
    operating_site_list.append(operating_site2)
    # operating_site = operating_site2

    for operating_site in operating_site_list:

        if operating_site == operating_site1:
            log_path = "/home/keti_iisrc/test/log/"
            logname = "operation1"
            logfile = logname + ".json"
        elif operating_site == operating_site2:
            log_path = "/home/keti_iisrc/operation2/log/"
            logname = "operation2"
            logfile = logname + ".json"

        dataloss_logger = logs.get_logger(logname, log_path, logfile)

        timescale_local = timescale_input_test.timescale(
            ip="1.214.41.250",
            port="5434",
            username="postgres",
            password="keti1234!",
            dbname=operating_site,
        )

        timescale_GCP = timescale_input_test.timescale(
            ip="localhost",
            port="5432",
            username="postgres",
            password="keti1234!",
            dbname=operating_site,
        )

        """1. 특정 날짜 지정"""
        # seoul = pytz.timezone("Asia/Seoul")
        # start_time = datetime(2022, 5, 30, 0, 0, 0)
        # start_time = seoul.localize(start_time)
        # end_time = datetime(2022, 6, 2, 0, 0, 0)
        # end_time = seoul.localize(end_time)

        # table_list = ["bank", "rack", "pcs", "etc"]

        # for table_name in table_list:
        #     print("local DB")
        #     print(operating_site)
        #     calc_loss_rate(
        #         timescale_local, operating_site, start_time, end_time, table_name
        #     )
        #     print("--------------")
        #     print(operating_site)
        #     print("GCP DB")
        #     calc_loss_rate(timescale_GCP, operating_site, start_time, end_time, table_name)
        #     print("--------------")

        """--------------------------------------------"""

        """2. 현재 날짜만 cron용 """
        end_time = datetime.now(timezone("asia/seoul")).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_time = end_time + timedelta(days=-1)

        table_list = ["bank", "rack", "pcs", "etc"]

        for table_name in table_list:
            print("local DB")
            print(operating_site)
            try:
                calc_loss_rate(
                    timescale_local, operating_site, start_time, end_time, table_name
                )
            except Exception as e:
                print(e)

            print("--------------")
            print(operating_site)
            print("GCP DB")

            try:
                calc_loss_rate(
                    timescale_GCP, operating_site, start_time, end_time, table_name
                )
            except Exception as e:
                print(e)

            print("--------------")
        end_time = end_time + timedelta(days=-1)

        # """--------------------------------------------"""

        """3. 특정날짜부터 현재까지 데이터 손실률 하루하루 """
        # seoul = pytz.timezone("Asia/Seoul")
        # start_time = datetime(2022, 5, 30, 0, 0, 0)
        # start_time = seoul.localize(start_time)
        # end_time = datetime.now(timezone("asia/seoul")).replace(
        #     hour=0, minute=0, second=0, microsecond=0
        # )
        # date_diff = end_time - start_time

        # for i in range(date_diff.days):
        #     start_time = end_time + timedelta(days=-1)
        #     table_list = ["bank", "rack", "pcs", "etc"]

        #     for table_name in table_list:
        #         print("local DB")
        #         print(operating_site)
        #         calc_loss_rate(
        #             timescale_local, operating_site, start_time, end_time, table_name
        #         )
        #         print("--------------")
        #         print(operating_site)
        #         print("GCP DB")
        #         calc_loss_rate(
        #             timescale_GCP, operating_site, start_time, end_time, table_name
        #         )
        #         print("--------------")
        #     end_time = end_time + timedelta(days=-1)
