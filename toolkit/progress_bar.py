from . import orakul_studio
from tqdm import tqdm
import time
import subprocess

_last_gpu_check = 0
_gpu_str = ""

def get_gpu_stats():
    global _last_gpu_check, _gpu_str
    cur_t = time.time()
    # Опрос каждые 5 секунд, чтобы не тормозить процесс
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

class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        # Дёргаем функцию из твоего отдельного файла строго при создании этого бара!
        orakul_studio.print_banner()
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)

    # Вот эта магия встраивает вольтаж прямо в движок ползунка
    @property
    def format_dict(self):
        d = super().format_dict
        stats = get_gpu_stats()
        if stats:
            d['prefix'] = f"{stats}{d.get('prefix', '')}"
        return d