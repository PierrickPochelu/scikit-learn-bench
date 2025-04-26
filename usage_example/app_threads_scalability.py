from scikit_learn_bench.all_bench import all_bench
from scikit_learn_bench.display import print_table
from scikit_learn_bench import CONST
import os
from joblib import parallel_backend
# https://scikit-learn.org/stable/computing/parallelism.html#parallel-numpy-and-scipy-routines-from-numerical-libraries

backend_name="loky" # threading or loky
num_jobs=2
num_threads_per_job=64

timesteps_score=[]
experiences = [(1,1), (num_jobs, num_threads_per_job)]
for exp in experiences:
    jobs, threads = exp
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["BLIS_NUM_THREADS"] = str(threads)

    with parallel_backend(backend_name, n_jobs=jobs):
        all_scores = all_bench(
            num_samples=100,
            num_features=100,
            num_output=2,
            min_prof_time=0.1,
            ml_type="all",
            profiler_type="time",
            table_print=False
        )
        timesteps_score.append(all_scores)

#### VIZUALISATION ###

scalability_score={}
for algo_name in timesteps_score[0].keys():
    variables=[]
    num_metrics=len(timesteps_score[0][algo_name])
    for i in range(num_metrics):
        if isinstance(timesteps_score[1][algo_name][i],str) or isinstance(timesteps_score[0][algo_name][i],str) :
            v = CONST.NANSTR
        else:
            v = timesteps_score[1][algo_name][i] / timesteps_score[0][algo_name][i]
        variables.append(v)
    scalability_score[algo_name]=variables

print_table(scalability_score, 1)