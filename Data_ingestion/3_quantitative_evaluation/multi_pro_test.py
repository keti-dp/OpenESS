import time
from multiprocessing import Pool
from multiprocessing import Process


start_time = time.time()


def count(process_name):
    for i in range(1, 100000):
        a = i + 1
        pass


class MyProcess(Process):
    def __init__(self, string):
        Process.__init__(self)
        self.string = string

    def run(self):
        count(self.string)


# process
if __name__ == "__main__":
    proc = MyProcess("process")
    proc.start()
    print(time.time() - start_time)
