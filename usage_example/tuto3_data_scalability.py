from src.core import bench
from src.display import print_table
from src import CONST



# Run benchmarks
categories = [
    ("clu", 2),
    ("tra", 1),
    ("cla", 2),
    ("reg", 1),
    ("reg", 2)
]

timesteps_score=[]

sample_sizes = [10, 100]
for num_samples in sample_sizes:

    all_scores = {}
    model_counts = {}

    for ml_category, num_output in categories:
        scores = bench(
            num_samples=num_samples,
            num_features=2,
            num_output=num_output,
            fix_comp_time=0.1,
            ml_type=ml_category,
            profiler_type="timememory",
            table_print=False
        )

        for model_name, result in scores.items():
            if model_name in model_counts:
                model_counts[model_name] += 1
                all_scores[f"{model_name}{model_counts[model_name]}"] = result
            else:
                model_counts[model_name] = 1
                all_scores[model_name] = result

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