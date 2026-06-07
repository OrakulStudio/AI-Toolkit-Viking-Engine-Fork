import sys
import os
import time
import subprocess
from toolkit.accelerator import get_accelerator

_last_gpu_check = 0
_gpu_str = ""

def get_gpu_stats():
    global _last_gpu_check, _gpu_str
    cur_t = time.time()
    if cur_t - _last_gpu_check > 5:
        try:
            res = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw,clocks.gr', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            ).strip().split(', ')
            _gpu_str = f"[{res[0]}W {res[1]}MHz] "
        except Exception:
            pass
        _last_gpu_check = cur_t
    return _gpu_str

def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        stats = get_gpu_stats()
        new_args = list(args)
        if new_args and isinstance(new_args[0], str) and stats:
            new_args[0] = f"{stats}{new_args[0]}"
        print(*new_args, **kwargs)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    sys.stdout = Logger(filename)
    sys.stderr = Logger(filename)