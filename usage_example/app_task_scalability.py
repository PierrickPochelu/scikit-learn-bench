from scikit_learn_bench.all_bench import all_bench
from scikit_learn_bench.display import print_table
from scikit_learn_bench import CONST

# Run benchmarks

timesteps_score=[]

candidate_values = [10, 100]
param_to_vary = "num_samples"

for candidate_value in candidate_values:
    kwargs={"num_samples":100,
        "num_features":100,
        "num_output":2,
        "min_prof_time":0.1,
        "ml_type":"all",
        "profiler_type":"timememory",
        "table_print":False}
    kwargs[param_to_vary]=candidate_value
    all_scores = all_bench(**kwargs)


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