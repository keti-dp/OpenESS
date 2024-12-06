# ESS 4차년도때 만든거 컨테이너 CPU 메모리 스레스홀드 텔레그램 보내기

"""
docker_monitoring.py : 도커 컨테이너 모니터링 시스템

        ---------------------------------------------------------------------------
        Copyright(C) 2023, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------

        Copyright(C) 2023, 윤태일 / KETI / taeil777@keti.re.kr
        ---------------------------------------------------------------------------
        이 프로그램은 자유 소프트웨어입니다. 당신은 자유 소프트웨어 재단이 공표한 GNU 일반 공중 라이선스 버전 2 또는 
        그 이후 버전을 임의로 선택해서 그 규정에 따라 프로그램을 수정하거나 재배포할 수 있습니다.

        이 프로그램은 유용하게 사용될 수 있을 것이라는 희망에서 배포되고 있지만 어떠한 형태의 보증도 제공하지 않습니다. 
        상품성 또는 특정 목적 적합성에 대한 묵시적 보증 역시 제공하지 않습니다. 보다 자세한 내용은 GNU 일반 공중 라이선스를 참고하시기 바랍니다.

        GNU 일반 공중 라이선스는 이 프로그램과 함께 제공됩니다. 만약, 라이선스를 받지 못했다면, 
        자유 소프트웨어 재단으로 문의하기 바랍니다. 
        주소: Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

도커 컨테이너 상태를 모니터링하고 메시지를 보내는 프로그램입니다.

컨테이너의 리소스를 모니터링하고 설정한 임계값에 따라 텔레그램으로 메시지를 보냅니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
       
"""


import docker
import telegram
import threading
import time
import asyncio

# 텔레그램 봇 토큰 및 채팅 ID 설정
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# 리소스 사용량 임계값 설정 (퍼센트)
CPU_THRESHOLD = 10
MEM_THRESHOLD = 10

# 이벤트 루프 생성
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


async def send_telegram_message(message):
    """텔레그램으로 메시지 전송"""
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)


def monitor_docker_events():
    """Docker 이벤트 모니터링"""
    client = docker.from_env()
    docker_api = client.api

    filters = {
        "event": ["start", "stop", "die", "health_status"],
        "type": "container",
    }

    events = docker_api.events(filters=filters, decode=True)

    for event in events:
        status = event.get("status")
        container_id = event.get("id")[:12]
        container_name = event["Actor"]["Attributes"].get("name", "Unknown")
        message = f"컨테이너 {container_name} ({container_id}) 상태 변경: {status}"
        asyncio.run(send_telegram_message(message))


def monitor_container_stats(container):
    """컨테이너 리소스 사용량 모니터링"""
    try:
        for stats in container.stats(decode=True):
            cpu_stats = stats["cpu_stats"]
            precpu_stats = stats["precpu_stats"]

            # 키 존재 여부 확인 및 초기 루프 건너뛰기
            if (
                not precpu_stats
                or "system_cpu_usage" not in cpu_stats
                or "system_cpu_usage" not in precpu_stats
            ):
                time.sleep(5)
                continue

            # CPU 사용량 계산
            cpu_delta = (
                cpu_stats["cpu_usage"]["total_usage"]
                - precpu_stats["cpu_usage"]["total_usage"]
            )
            system_cpu_delta = (
                cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]
            )
            if system_cpu_delta > 0.0 and cpu_delta > 0.0:
                cpu_percent = (
                    (cpu_delta / system_cpu_delta)
                    * len(cpu_stats["cpu_usage"].get("percpu_usage", []))
                    * 100.0
                )
            else:
                cpu_percent = 0.0

            # 메모리 사용량 계산 (이전과 동일)
            mem_usage = stats["memory_stats"]["usage"] - stats["memory_stats"][
                "stats"
            ].get("cache", 0)
            mem_limit = stats["memory_stats"]["limit"]
            mem_percent = (mem_usage / mem_limit) * 100.0

            # 임계값 초과 시 텔레그램 알림 전송
            if cpu_percent > CPU_THRESHOLD:
                message = (
                    f"컨테이너 {container.name} CPU 사용량 초과: {cpu_percent:.2f}%"
                )
                asyncio.run(send_telegram_message(message))

            if mem_percent > MEM_THRESHOLD:
                message = (
                    f"컨테이너 {container.name} 메모리 사용량 초과: {mem_percent:.2f}%"
                )
                asyncio.run(send_telegram_message(message))

            time.sleep(5)
    except docker.errors.APIError as e:
        message = f"컨테이너 {container.name} 리소스 모니터링 중 오류 발생: {e}"
        asyncio.run(send_telegram_message(message))
    except Exception as e:
        message = f"컨테이너 {container.name} 모니터링 중 예기치 않은 오류 발생: {e}"
        asyncio.run(send_telegram_message(message))


def start_monitoring_all_containers():
    """모든 컨테이너에 대해 리소스 모니터링 스레드 시작"""
    client = docker.from_env()
    monitored_containers = set()

    while True:
        current_containers = set(client.containers.list())
        new_containers = current_containers - monitored_containers

        for container in new_containers:
            threading.Thread(
                target=monitor_container_stats, args=(container,), daemon=True
            ).start()
            monitored_containers.add(container)

        # 제거된 컨테이너 업데이트
        monitored_containers = monitored_containers & current_containers
        time.sleep(10)  # 새로운 컨테이너 체크 주기


if __name__ == "__main__":
    # 이벤트 루프 실행
    threading.Thread(target=loop.run_forever, daemon=True).start()

    # Docker 이벤트 모니터링 스레드 시작
    threading.Thread(target=monitor_docker_events, daemon=True).start()

    # 모든 컨테이너의 리소스 모니터링 스레드 시작
    threading.Thread(target=start_monitoring_all_containers, daemon=True).start()

    # 메인 스레드 유지
    while True:
        time.sleep(1)
