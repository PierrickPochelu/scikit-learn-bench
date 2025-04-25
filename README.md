# scikit-learn-bench

Benchmark **100+ scikit-learn Machine Learning algorithms** at once — in just a few seconds.

This tool offers an easy way to evaluate models across multiple ML categories and profiling strategies.

---

## 📦 Installation

```
pip3 install scikit_learn_bench
```

## 🚀 Features

### 📊 Data Generation

Easily control the characteristics of synthetic datasets:

- `num_samples`: Number of samples (rows)
- `num_features`: Number of input features (columns)
- `num_output`: Target shape — used for regression, classification, clusters, or transformed outputs

---

### 🧠 ML Algorithm Types

| Type            | Label | Description                                                       |
|-----------------|-------|-------------------------------------------------------------------|
| Regressors      | `"reg"` | 55 algorithms with 2 methods `.fit` and `.predict`                |
| Classifiers     | `"cla"` | 42 algorithms with 2 methods `.fit` and `.predict`                |
| Clustering      | `"clu"` | 12 clustering algorithms (`.predict` supported for 6)             |
| Transformations | `"tra"` | 68 transform functions (e.g. `MinMaxScaler`, `PCA`, `TSNE`, etc.) |

In total, the tool allows benchmarking 280 scikit-learn functions (55\*2+42\*2+12+6+68).

> The exact counts may vary depending on your installed `scikit-learn` version (here 1.6.1), the data characteristics (#samples, #features, #output), and other software dependencies.

---

### ⏱ Profiling Strategies

Choose one of three profiler types:

- `"time"`:  
  Measures **training and inference throughput** (samples/sec).  
  Output: `(train_throughput, infer_throughput)`

- `"timememory"`:  
  Adds **peak memory** (kB) with `tracemalloc`.  
  Output: `(train_throughput, infer_throughput, train_peak_memory, infer_peak_memory)`

- `"timeline"`:  
  Fine-grained `cProfile` analysis saved as `.prof` files for each algorithm.  
  Output: `.prof` file per model

---

### ⚙️ Other Parameters

- `min_prof_time`: Minimum time in seconds to run the profiling of each function (reduces noise)
- `max_prof_time`: Maximum time in seconds to run the profiling. If the delay is reached, a timeout interrupts the computation.
- `table_print`: Display formatted results in console
- `table_print_sort_crit`: Sort results (e.g., by training speed)
- `line_profiler_path`: Path to store `.prof` files for `"timeline"` profiler

---

## 🧪 Example Usage



### CLI

After installing `scikit_learn_bench`, you can invoke it directly from the command line:

```commandline
pierrick@laptop:~$ pip3 install scikit_learn_bench
pierrick@laptop:~$ scikit_learn_bench
```

For detailed usage and options, run:

```commandline
scikit_learn_bench --help
```

### Programming interface (advanced)

```python
from scikit_learn_bench.core import bench

scores = bench(
  num_samples=10,
  num_features=2,
  num_output=2,
  min_prof_time=0.1,
  ml_type="cla",
  profiler_type="timememory",
  table_print=True
)
```

This function returns a dictionary with performance metrics for each algorithm, such as:
```
{
    'AdaBoostClassifier': (4454.128, 48093.051, 94.06, 19.29),
    'BaggingClassifier': (282.16, 6696.015, 96.019, 162.843),
    ...
}
```

The output includes:
* Train/s: Training speed (samples per second)
* Train Mem: Memory usage during training (MB)
* Infer/s: Inference speed (samples per second)
* Infer Mem: Memory usage during inference (MB)

### Advanced performance analysis

dAditionally, the `usage_example/` directory contains scripts for advanced analyses, including:

* 2D cloud points comparing throughput and memory consumption across all algorithms
* Scalability studies examining how performance varies with data size (samples, features, output size)
* Analysis of algorithm performance as the number of CPU cores increases, helping identify which algorithms benefit most from parallel processing



## 📚 Citing scikit-learn-bench
Work in progress: BibTeX / reference for citing this repo.

## 🙏 Acknowledgments

ULHPC Platform for computing support and motivating this project.
