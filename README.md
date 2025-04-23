# scikit-learn-bench

Benchmark **150+ scikit-learn Machine Learning algorithms** at once — in just a few seconds.

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

| Type | Label | Description |
|------|-------|-------------|
| Regressors | `"reg"` | 53 algorithms with `.fit` and `.predict` |
| Classifiers | `"cla"` | 41 algorithms with `.fit` and `.predict` |
| Clustering | `"clu"` | 12 clustering algorithms (`predict` supported for 6) |
| Transformers | `"tra"` | 56 preprocessing/transform models (e.g. `MinMaxScaler`, `PCA`, `TSNE`, etc.) |

> The exact counts may vary depending on your installed `scikit-learn` version.

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

```python
bench(
    num_samples=10,
    num_features=2,
    num_output=num_output,
    fix_comp_time=0.1,
    ml_type=ml_category,
    profiler_type="timememory",
    table_print=True
)
```

Sample Output:

```
Algorithm                     Train/s   Train Mem  Infer/s  Infer Mem
----------------------------------------------------------------------
RandomForestClassifier         41.5     1674       92.6     78.8
ExtraTreesClassifier           69.7     2146       89.0     122
StackingClassifier             125      1463       222      11.4
[...]
NearestCentroid                29314    14804      10.0     13.8
KNeighborsClassifier           30762    23863      18.4     27.0
DummyClassifier                82341    540676     11.6     57.6
```

## 📦 Installation
Work in progress: Incoming pip install instructions.

## 📚 Citing scikit-learn-bench
Work in progress: BibTeX / reference for citing this repo.

## 🙏 Acknowledgments

ULHPC Platform for computing support and motivating this project.
