import time
from collections import OrderedDict, deque
import sys
import os

# check if is ui process will have IS_AI_TOOLKIT_UI in env
is_ui = os.environ.get("IS_AI_TOOLKIT_UI", "0") == "1"

class Timer:
    def __init__(self, name='Timer', max_buffer=10):
        self.name = name
        self.max_buffer = max_buffer
        self.timers = OrderedDict()
        self.active_timers = {}
        self.current_timer = None  # Used for the context manager functionality
        self._after_print_hooks = []

    def start(self, timer_name):
        # Полностью глушим начало замера времени, чтобы процессор не отвлекался
        return

    def cancel(self, timer_name):
        """Cancel an active timer."""
        if timer_name in self.active_timers:
            del self.active_timers[timer_name]

    def stop(self, timer_name):
        # Полностью глушим остановку и запись в массивы, освобождаем шину
        return

    def add_after_print_hook(self, hook):
        self._after_print_hooks.append(hook)

    def print(self):
        # Прокидываем пустой словарь в хуки движка, чтобы ничего не ломалось,
        # и мгновенно выходим, полностью заблокировав вывод простыни на экран
        for hook in self._after_print_hooks:
            hook({})
        return

    def reset(self):
        self.timers.clear()
        self.active_timers.clear()

    def __call__(self, timer_name):
        """Enable the use of the Timer class as a context manager."""
        self.current_timer = timer_name
        self.start(timer_name)
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # No exceptions, stop the timer normally
            self.stop(self.current_timer)
        else:
            # There was an exception, cancel the timer
            self.cancel(self.current_timer)