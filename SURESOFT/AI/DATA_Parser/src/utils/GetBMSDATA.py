import serial
import time
import datetime
import threading
import os


class BMS:
    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        record: bool = True,
        visual: bool = False,
        pipeline=None,
    ):
        threading.Thread().__init__(self)
        # Default Setting
        self.record = self.do_nothing
        self.visual = self.do_nothing
        self.pipeline = self.do_nothing

        # Define Function according Options
        self.record_status = record
        if self.record_status:
            # Make Record txt file
            self.make_record_file()
            self.record = self.record_data
        if visual:
            self.visual = self.show_log
        if pipeline is not None:
            self.pipeline = pipeline.send
        self.sensor = self.connect_sensor(port, baudrate, bytesize, parity, stopbits)

    def do_nothing(self, *args):
        return

    def connect_sensor(self, port, baudrate, bytesize, parity, stopbits):
        while True:
            try:
                sensor = serial.Serial(
                    port=port,
                    baudrate=baudrate,
                    bytesize=bytesize,
                    parity=parity,
                    stopbits=stopbits,
                )
                self.status = "Connected"
                sensor.flushInput()
                sensor.flushInput()
                return sensor
            except ConnectionError as e:
                self.status = "ConnectionError"
                print(e)
                time.sleep(1)

    def make_record_file(self):
        self.file_name = (
            os.getcwd()
            + "/output/"
            + datetime.datetime.strftime(datetime.datetime.now(), format="%Y%m%d_%H%M")
            + "_BMS.txt"
        )
        if not os.path.isdir(os.getcwd() + "/output/"):
            os.mkdir(os.getcwd() + "/output/")
        savefile = open(self.file_name, "w")
        savefile.write(
            "Time,CPU_usage,stack_usage,heap_current,heap_max,temperature,timestamp,\n"
        )
        savefile.close()

        print("GPS Data Record Start!")

    def record_data(self):
        with open(self.file_name, "a") as self.savefile:
            self.savefile.write(
                f"{str(self.time)},\
                                {str(self.CPU_usage)},\
                                {str(self.stack_usage)},\
                                {str(self.heap_current)},\
                                {str(self.heap_max)},\
                                {str(self.temperature)},\
                                {str(self.timestamp)}\n"
            )

    def show_log(self):
        print(self.log)

    def read_signal(self):
        """
        read one line of GPS signal
        """
        try:
            self.log = self.sensor.readline().decode("utf-8")
        except ConnectionError as e:
            self.status = "ConnectionError"
            print(e)
            self.sensor = self.connect_sensor(self.port_num, self.bandrate)

            return

        self.log = self.log.split(",")
        self.status = "NormalData"
        self.time = datetime.datetime.now().isoformat()
        self.CPU_usage = float(self.log[0])
        self.stack_usage = float(self.log[1])
        self.heap_current = float(self.log[2])
        self.heap_max = float(self.log[3])
        self.temperature = float(self.log[4])
        self.timestamp = float(self.log[5])

        # 데이터 저장
        self.record()
        # # 데이터 출력
        self.visual()

    def run(self):
        while True:
            self.read_signal()


if __name__ == "__main__":
    sensor = BMS(record=False, visual=True)
    while True:
        sensor.read_signal()
