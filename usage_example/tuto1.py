from src.core import bench
from src.display import print_table
from src import CONST

all_scores = {}

# Run benchmarks
categories = [
    ("clu", 2),
    ("tra", 1),
    ("cla", 2),
    ("reg", 1),
    ("reg", 2)
]

for ml_category, num_output in categories:
    scores = bench(
        num_samples=10,
        num_features=2,
        num_output=num_output,
        fix_comp_time=0.1,
        ml_type=ml_category,
        profiler_type="time",
        table_print=True
    )

    num_collected_algos=0
    for model_name, result in scores.items():
        if model_name in all_scores:
            pass # already evaluated
        else:
            all_scores[model_name] = result
            num_collected_algos += 1
    print(f"Num collected algos in categ. '{ml_category}': {num_collected_algos}")

print("Number of ML algo retrieved: ", len(all_scores))
print("Number of ML algo inference retrieved: ", len([s[1] for s in all_scores if s!=CONST.NANSTR]))
print("Performance collected:")
print_table(all_scores)

