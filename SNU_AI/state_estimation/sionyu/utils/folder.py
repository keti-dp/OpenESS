import os

def write_log(log_file, str):
    with open(log_file, 'a') as f:
        f.write(str)