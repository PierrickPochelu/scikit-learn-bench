import multiprocessing
import time
import os
import signal

from scikit_learn_bench import CONST

def timeout_warp(func, timeout):
    def wrapper(*args, **kwargs):
        output_queue = multiprocessing.Queue()

        def target():
            """this function manage the input/output and kill hierarchically"""
            os.setpgrp()  # all children will belong to the same group of processes
            result = func(*args, **kwargs)
            output_queue.put(result)

        proc = multiprocessing.Process(target=target)
        proc.start()
        #print(f"Launch {func.__self__} {args} timeout:", timeout)
        proc.join(timeout=timeout) # warning: join in debug mode will always wait `timeout` even if not necessary
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass # it is possible that the process is already killed
        if proc.is_alive():
            proc.terminate()
            proc.join()



        if not output_queue.empty():
            return output_queue.get()
        else:
            return CONST.NANSTR

    return wrapper