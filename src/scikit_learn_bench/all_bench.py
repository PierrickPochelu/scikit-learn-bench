from src.scikit_learn_bench.core import bench
from src.scikit_learn_bench.display import print_table
from src.scikit_learn_bench import CONST


def all_bench(num_samples: int = 100,
          num_features: int = 100,
          num_output: int = 2,
          fix_comp_time: float = 0.1,
          ml_type: str = "cla", # ignored
          profiler_type: str = "time",
          table_print: bool = True,
          table_print_sort_crit: int = 1,
          line_profiler_path:str = "."):

    all_scores = {}

    # Run benchmarks
    categories = [
        ("clu", num_output),
        ("tra", num_output),
        ("tra", 1),
        ("cla", num_output),
        ("reg", num_output),
        ("reg", 1)
    ]

    for ml_category, num_output in categories:
        scores = bench(
            num_samples=num_samples,
            num_features=num_features,
            num_output=num_output,
            fix_comp_time=fix_comp_time,
            ml_type=ml_category,
            profiler_type=profiler_type,
            table_print=False, # not individually display
            line_profiler_path=line_profiler_path
        )

        num_collected_algos=0
        for model_name, result in scores.items():
            if model_name in all_scores:
                pass # already evaluated
            else:
                all_scores[model_name] = result
                num_collected_algos += 1
        print(f"Num collected algos in categ. '{ml_category} output size:{num_output}': {num_collected_algos}")

    print("Number of ML algo retrieved: ", len(all_scores))
    print("Performance collected:")
    print_table(all_scores, table_print_sort_crit)
