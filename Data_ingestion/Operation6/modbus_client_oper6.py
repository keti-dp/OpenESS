"""
modbus_client_oper6.py : 태양광 ESS 데이터 수집을 위한 코드

        ---------------------------------------------------------------------------
        Copyright(C) 2025, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------

        ---------------------------------------------------------------------------
        The MIT License

        Copyright(C) 2025, 윤태일 / KETI / taeil777@keti.re.kr

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

석환 태양광 ESS데이터 수집을 위한 코드입니다.

ModbusTCP 통신에 의해 1초 단위로 데이터가 수집되며

파싱해온 데이터를 저장규격에 맞게 필터링하고 재가공하여

GCP 인스턴스에 구축된 Timescale DB에 저장합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.


"""

import asyncio
from pyModbusTCP.client import ModbusClient
from datetime import datetime
from pytz import timezone
import asyncpg


MODBUS_HOST = ""
MODBUS_PORT = ""
DB_DSN = "postgresql://postgres:"


# -------------------------------------------------------------------
# 모드버스 다량 레지스터 안전하게 읽어오기 위한 함수 (최대 125개 제한 보완)
# -------------------------------------------------------------------
async def read_modbus_range(client, start_addr, end_addr):
    loop = asyncio.get_event_loop()
    result = []
    total_count = end_addr - start_addr + 1
    MAX_REG = 125

    quotient, remainder = divmod(total_count, MAX_REG)
    current_addr = start_addr

    for _ in range(quotient):
        reg_list = await loop.run_in_executor(
            None, client.read_input_registers, current_addr, MAX_REG
        )
        if reg_list is None:
            reg_list = []
        result += reg_list
        current_addr += MAX_REG

    if remainder > 0:
        reg_list = await loop.run_in_executor(
            None, client.read_input_registers, current_addr, remainder
        )
        if reg_list is None:
            reg_list = []
        result += reg_list

    return result


# 음수변형 2의보수
def twos_complement(v: int) -> int:
    return v - 65536 if v > 32767 else v


# -------------------------------------------------------------------
# Bank 데이터 (1개 Bank) 읽기 예시
# -------------------------------------------------------------------
async def get_bank_data(client):
    # 두의 보수 변환 함수: 0~32767 그대로, 그 이상은 v-65536

    # 예: 주소 0 ~ 12, 1000 ~ 1045
    bank_part1 = await read_modbus_range(client, 0, 12)
    bank_part2 = await read_modbus_range(client, 1000, 1045)

    # 각 레지스터에 곱할 스케일링 계수 (0~12 총 13개)
    multipliers = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.001,
        0.001,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    # Part1: 먼저 두의 보수 적용 → 그다음 스케일링
    part1_processed = [
        round(twos_complement(raw) * mul, 3)
        for raw, mul in zip(bank_part1, multipliers)
    ]

    # Part2: 두의 보수만 적용
    part2_processed = [twos_complement(v) for v in bank_part2]

    bank_raw = part1_processed + part2_processed
    bank_data = {"bank_id": 1, "values": bank_raw}

    # bank_id 맨앞에 추가
    # return bank_data
    return [1] + part1_processed + part2_processed


# -------------------------------------------------------------------
# Rack 데이터 (9개 Rack) 읽기
# - 첫 구간(13~219)을 9등분
# - 두 번째 구간(1046~1477)을 9등분
# - 같은 rack_index에서 가져온 part1/part2를 합쳐서 하나의 rack 정보로 구성
# -------------------------------------------------------------------
async def get_rack_data(client):
    # 두의 보수 변환 함수

    # 첫 번째 구간: 13 ~ 219
    rack_part1 = await read_modbus_range(client, 13, 219)
    # 두 번째 구간: 1046 ~ 1477
    rack_part2 = await read_modbus_range(client, 1046, 1477)

    rack_count = 9
    part1_len = len(rack_part1)
    part2_len = len(rack_part2)
    part1_rack_size = part1_len // rack_count
    part2_rack_size = part2_len // rack_count

    multipliers = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.001,
        1,
        0.001,
        1,
        0.001,
        0.001,
        0.1,
        1,
        0.1,
        1,
        0.1,
        0.1,
        1,
        0.1,
        1,
        0.1,
        1,
        0.1,
        1,
    ]

    # part2는 두의 보수만 미리 적용
    part2_processed = [twos_complement(v) for v in rack_part2]

    racks = []
    for i in range(rack_count):
        # part1 slice
        start1 = i * part1_rack_size
        end1 = start1 + part1_rack_size
        slice1 = rack_part1[start1:end1]

        # 두의 보수 → 스케일링 → 반올림
        processed1 = [
            round(twos_complement(val) * multipliers[idx], 3)
            for idx, val in enumerate(slice1)
        ]

        # part2 slice
        start2 = i * part2_rack_size
        end2 = start2 + part2_rack_size
        slice2 = part2_processed[start2:end2]

        combined = processed1 + slice2
        # [bank_id, rack_id, *values] 형태로 리스트 반환
        racks.append([1, i + 1, *combined])

    return racks


# -------------------------------------------------------------------
# Module 데이터 읽기 (Rack 9개, Rack당 Module 17개, Module마다 12개 셀)
# -------------------------------------------------------------------
async def get_module_data(client):
    # 두의 보수 변환 함수: 0~32767 그대로, 그 이상은 v-65536

    # 예: 주소 5000 ~ 6835
    module_raw = await read_modbus_range(client, 5000, 6835)

    # 1) 두의 보수 적용
    # 2) 0.0001 곱하기
    # 3) 소수점 1자리로 반올림
    # scaled_raw = [round(twos_complement(v) * 0.0001, 1) for v in module_raw]
    scaled_raw = [round(v * 0.0001, 3) for v in module_raw]

    rack_count = 9
    modules_per_rack = 17
    cells_per_module = 12

    total_cells = rack_count * modules_per_rack * cells_per_module
    if len(scaled_raw) < total_cells:
        print(f"모듈 데이터가 부족합니다. (읽은 레지스터 수: {len(scaled_raw)})")
        return []

    modules = []
    idx = 0

    for rack_id in range(1, rack_count + 1):
        for module_id in range(1, modules_per_rack + 1):
            cell_values = scaled_raw[idx : idx + cells_per_module]
            idx += cells_per_module

            # [bank_id, rack_id, module_id, *cell_values]
            modules.append([1, rack_id, module_id, *cell_values])

    return modules


async def get_pcs_data(client):

    # PCS 데이터 읽기 (주소 2000 ~ 2053)
    pcs_raw = await read_modbus_range(client, 2000, 2053)

    # 2000번부터 2019번까지 적용할 스케일링 계수 (총 20개)
    multipliers = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.01,
        0.1,
        1,
        0.1,
        1,
        1,
        1,
        1,
        1,
        0.01,
        0.1,
    ]

    processed = []
    for idx, rawv in enumerate(pcs_raw):
        # 두의 보수 변환
        v = twos_complement(rawv)
        # 2000~2019번(인덱스 0~19)에만 스케일링 + 소수점 첫째 자리 반올림
        if idx < len(multipliers):
            v = round(v * multipliers[idx], 1)
        processed.append(v)

    return processed


async def get_etc_data(client):

    # ETC데이터 읽기 (주소 3000 ~ 3005)
    etc_raw = await read_modbus_range(client, 3000, 3005)

    # 2000번부터 2019번까지 적용할 스케일링 계수 (총 20개)
    multipliers = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.001,
    ]

    etc_data = []
    for idx, rawv in enumerate(etc_raw):
        # 두의 보수 변환
        v = twos_complement(rawv)

        if idx < len(multipliers):
            v = round(v * multipliers[idx], 1)
        etc_data.append(v)

    return etc_data


async def periodic_modbus():
    client = ModbusClient(host=MODBUS_HOST, port=MODBUS_PORT, unit_id=1, auto_open=True)
    conn = await asyncpg.connect(DB_DSN)

    # 1) 최초 실행을 바로 다음 5초 경계(00,05,10…)로 맞춤
    now = datetime.now(timezone("Asia/Seoul"))
    sec = now.second % 5
    if sec != 0:
        await asyncio.sleep(5 - sec)

    try:
        while True:
            now = datetime.now(timezone("Asia/Seoul")).replace(microsecond=0)
            print(f"[{now}] 모드버스 데이터 읽기 시작")

            try:
                # 1) Bank
                bank_list = await get_bank_data(client)
                args = [now, *bank_list]
                ph = ",".join(f"${i}" for i in range(1, len(args) + 1))
                await conn.execute(f"INSERT INTO bank VALUES({ph})", *args)
                print(f"[{now}] Bank data inserted.")

                # 2) Rack
                rack_lists = await get_rack_data(client)
                for rl in rack_lists:
                    args = [now, *rl]
                    ph = ",".join(f"${i}" for i in range(1, len(args) + 1))
                    await conn.execute(f"INSERT INTO rack VALUES({ph})", *args)
                print(f"[{now}] {len(rack_lists)} Rack records inserted.")

                # 3) Module
                module_lists = await get_module_data(client)
                for ml in module_lists:
                    args = [now, *ml]
                    ph = ",".join(f"${i}" for i in range(1, len(args) + 1))
                    await conn.execute(f"INSERT INTO module VALUES({ph})", *args)
                print(f"[{now}] {len(module_lists)} Module records inserted.")

                # 4) PCS
                pcs_list = await get_pcs_data(client)
                args = [now, *pcs_list]
                ph = ",".join(f"${i}" for i in range(1, len(args) + 1))
                await conn.execute(f"INSERT INTO pcs VALUES({ph})", *args)
                print(f"[{now}] PCS data inserted.")

                # 5) ETC
                etc_list = await get_etc_data(client)
                args = [now, *etc_list]
                ph = ",".join(f"${i}" for i in range(1, len(args) + 1))
                await conn.execute(f"INSERT INTO etc VALUES({ph})", *args)
                print(f"[{now}] ETC data inserted.")

            except Exception as e:
                print("데이터 읽기 중 오류 발생:", e)

            # 2) 다음 5초 경계까지 남은 시간만큼 sleep
            now2 = datetime.now(timezone("Asia/Seoul"))
            sec2 = now2.second % 5
            # 만약 정확히 경계에 와 있다면 full 5초 대기
            delay = 5 - sec2 if sec2 != 0 else 5
            await asyncio.sleep(delay)
    finally:
        await conn.close()


async def main():
    # 백그라운드로 주기 작업 등록
    asyncio.create_task(periodic_modbus())
    # 메인 루프가 종료되지 않도록 대기
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
