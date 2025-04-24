import os
import time
import tracemalloc
import cProfile
import pstats
from functools import wraps
from typing import Any, Callable, Tuple

from scikit_learn_bench import CONST


# ---------------------------
# Strategy Base
# ---------------------------
class ProfilerStrategy:
    def profile(self, func: Callable, T: float, *func_args) -> Any:
        raise NotImplementedError("Must override in subclass")

# ---------------------------
# Time-Only Profiler
# ---------------------------
class TimeProfiler(ProfilerStrategy):
    def profile(self, func: Callable, T: float, *func_args) -> float:
        time_count = 0
        start_time = time.time()

        while time.time() < start_time + T:
            func(*func_args)
            time_count += 1

        elapsed = time.time() - start_time
        data_chunk_size = len(func_args[0])
        samples_per_sec = (data_chunk_size * time_count) / elapsed if elapsed else 0
        return round(samples_per_sec, CONST.ROUNDING)

# ---------------------------
# Time + Memory Profiler (tracemalloc)
# ---------------------------
class TimeMemoryProfiler(ProfilerStrategy):
    def __init__(self):
        self.time_profiler = TimeProfiler()

    def profile(self, func: Callable, T: float, *func_args) -> Tuple[float, float]:
        tracemalloc.start()
        throughput = self.time_profiler.profile(func, T, *func_args)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return throughput, round(peak_mem / 1024, CONST.ROUNDING)  # KB

# ---------------------------
# cProfile Decorator Wrapper
# ---------------------------
def Profiling(output_file="profile.prof", sort_by="time", print_stats=100):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            profiler.dump_stats(output_file)
            stats = pstats.Stats(output_file)
            stats.strip_dirs().sort_stats(sort_by).print_stats(print_stats)
            return result
        return wrapper
    return decorator

# ---------------------------
# Time + cProfile Timeline Profiler
# ---------------------------
class TimeLineProfiler(ProfilerStrategy):
    def __init__(self, profile_folder_path: str = "./profiles"):
        self.time_profiler = TimeProfiler()
        self.profile_folder_path = profile_folder_path
        os.makedirs(self.profile_folder_path, exist_ok=True)

    def _build_filename(self, func: Callable, *args) -> str:
        name = func.__self__.__class__.__name__
        #arg_sig = "_".join([arg for arg in args])
        #filename = f"{name}_{arg_sig}.prof"
        filename = f"{name}.prof"
        return os.path.join(self.profile_folder_path, filename)

    def profile(self, func: Callable, T: float, *func_args) -> float:
        file_path = self._build_filename(func, *func_args)
        profiler_decorator = Profiling(output_file=file_path, sort_by="time", print_stats=20)

        @profiler_decorator
        def _profiled_call():
            return self.time_profiler.profile(func, T, *func_args)

        return _profiled_call()
