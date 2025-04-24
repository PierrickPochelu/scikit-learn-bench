import argparse
from scikit_learn_bench.core import bench
from scikit_learn_bench.all_bench import all_bench

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark scikit-learn models for training/inference time with profiling support."
    )

    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of training samples (default: 100)')
    parser.add_argument('--num-features', type=int, default=100,
                        help='Number of features (default: 100)')
    parser.add_argument('--num-output', type=int, default=2,
                        help='Number of output classes/targets/clusters (default: 2)')
    parser.add_argument('--fix-comp-time', type=float, default=0.1,
                        help='Max time (seconds) allowed for training each model (default: 1.0)')
    parser.add_argument('--ml-type', type=str, choices=['cla', 'reg', 'clu', 'tra', 'all'], default='all',
                        help="Type of ML task: 'cla', 'reg', 'clu', 'tra', 'all' (default: 'all')")
    parser.add_argument('--profiler-type', type=str, choices=['time', 'timememory', 'timeline'], default='time',
                        help="Profiler type: 'time', 'timememory', or 'timeline' (default: 'time')")
    #parser.add_argument('--no-table-print', action='store_false', dest='table_print',
    #                    help='Disable formatted table print (default: enabled)')
    parser.add_argument('--table-print-sort-crit', type=int, default=1,
                        help='Column index to sort results table by (default: 1)')
    parser.add_argument('--line-profiler-path', type=str, default='.',
                        help='Path to save line profiler output if using "timeline" profiler (default: current directory)')

    args = parser.parse_args()

    if args.ml_type=="all":
        all_bench(
            num_samples=args.num_samples,
            num_features=args.num_features,
            num_output=args.num_output,
            fix_comp_time=args.fix_comp_time,
            ml_type=args.ml_type,
            profiler_type=args.profiler_type,
            table_print=True,
            table_print_sort_crit=args.table_print_sort_crit,
            line_profiler_path=args.line_profiler_path
        )
    else:
        bench(
            num_samples=args.num_samples,
            num_features=args.num_features,
            num_output=args.num_output,
            fix_comp_time=args.fix_comp_time,
            ml_type=args.ml_type,
            profiler_type=args.profiler_type,
            table_print=True,
            table_print_sort_crit=args.table_print_sort_crit,
            line_profiler_path=args.line_profiler_path
        )