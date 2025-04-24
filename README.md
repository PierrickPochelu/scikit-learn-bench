# scikit-learn-bench

Benchmark **160 scikit-learn Machine Learning algorithms** at once — in just a few seconds.

This tool offers an easy way to evaluate models across multiple ML categories and profiling strategies.

---

## 🚀 Features

### 📊 Data Generation

Easily control the characteristics of synthetic datasets:

- `num_samples`: Number of samples (rows)
- `num_features`: Number of input features (columns)
- `num_output`: Target shape — used for regression, classification, clusters, or transformed outputs

---

### 🧠 ML Algorithm Types

| Type | Label | Description                                                       |
|------|-------|-------------------------------------------------------------------|
| Regressors | `"reg"` | 51 algorithms with `.fit` and `.predict`                          |
| Classifiers | `"cla"` | 41 algorithms with `.fit` and `.predict`                          |
| Clustering | `"clu"` | 12 clustering algorithms (`predict` supported for 6)              |
| Transformers | `"tra"` | 57 transform functions (e.g. `MinMaxScaler`, `PCA`, `TSNE`, etc.) |


> Some algorithms are callable in some specific conditions. 29 regressors manages multiple target regressions, 22 regressors are scalar only.

> The exact counts may vary depending on your installed `scikit-learn` version (here 1.3.0) and other dependencies.


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

- `fix_comp_time`: Minimum time in seconds to run each profile (reduces noise)
- `table_print`: Display formatted results in console
- `table_print_sort_crit`: Sort results (e.g., by training speed)
- `line_profiler_path`: Path to store `.prof` files for `"timeline"` profiler

---

## 🧪 Example Usage

### CLI

Installing scikit_learn_bench add it in your path and can be directly called

```commandline
pierrick@laptop:~$ pip3 install ./dist/scikit_learn_bench-0.0.1.tar.gz
pierrick@laptop:~$ scikit_learn_bench
```

After 4 seconds, output:



For more information: `scikit_learn_bench --help`



### Programming interface (advanced)
```python
from scikit_learn_bench.core import bench
scores=bench(
    num_samples=10,
    num_features=2,
    num_output=2,
    fix_comp_time=0.1,
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

## 📦 Installation
Work in progress: Incoming pip install instructions.

## 📚 Citing scikit-learn-bench
Work in progress: BibTeX / reference for citing this repo.

## 🙏 Acknowledgments

ULHPC Platform for computing support and motivating this project.
