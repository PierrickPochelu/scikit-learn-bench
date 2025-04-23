from src.core import bench

bench(
        num_samples=10,
        num_features=2,
        num_output=2,
        fix_comp_time=0.1,
        ml_type="cla",
        profiler_type="timememory",
        table_print=True,
        table_print_sort_crit=1
    )