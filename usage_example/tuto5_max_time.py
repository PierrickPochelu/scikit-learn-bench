import time

from scikit_learn_bench.core import bench
from scikit_learn_bench.display import print_table
from scikit_learn_bench import CONST

profiler_type = "time"
CONST.IS_MAX_PROF_TIME=True # activate
s = time.time()
scores = bench(num_samples=10, num_features=2, num_output=2, max_prof_time=0.1, min_prof_time=60, ml_type="clu",
               profiler_type=profiler_type)
enlapsed_time = time.time() - s
print("Profiling time: ",enlapsed_time)