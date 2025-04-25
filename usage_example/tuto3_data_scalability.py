from scikit_learn_bench.all_bench import all_bench
from scikit_learn_bench.display import print_table
from scikit_learn_bench import CONST

# Run benchmarks

timesteps_score=[]

sample_sizes = [10, 100]
for num_samples in sample_sizes:

    all_scores = all_bench(
        num_samples=num_samples,
        num_features=2,
        num_output=2,
        min_prof_time=0.1,
        ml_type="all",
        profiler_type="timememory",
        table_print=False
    )


    timesteps_score.append(all_scores)

####

scalability_score={}
for algo_name in timesteps_score[0].keys():
    variables=[]
    for i in range(4):
        if isinstance(timesteps_score[1][algo_name][i],str):
            v = CONST.NANSTR
        else:
            v = timesteps_score[1][algo_name][i] / timesteps_score[0][algo_name][i]
        variables.append(v)
    scalability_score[algo_name]=variables

print_table(scalability_score, 1)