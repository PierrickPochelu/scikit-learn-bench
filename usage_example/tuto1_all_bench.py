from scikit_learn_bench.all_bench import all_bench
all_bench(
        num_samples=10,
        num_features=2,
        num_output=2,
        min_prof_time=0.1,
        max_prof_time=60.,
        ml_type="all",
        profiler_type="time",
        table_print=True
    )
