"""
Numpy-style Documentation for the `bench` Module

This module provides functionality to benchmark the training and inference times
of various scikit-learn machine learning models. It supports classification,
regression, clustering, and transformation models and can optionally report
memory usage during execution.

Functions
---------
train_model(reg, X_train, y_train, T, score)
    Train and evaluate the performance of a machine learning model.

bench(num_samples, num_features, min_prof_time, max_prof_time, reg_or_cls="reg", nb_output=1)
    Benchmark training and inference times of scikit-learn models across categories.
"""
import numpy as np
from typing import Callable, Dict, Tuple, Union, List
import concurrent.futures

from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin, ClusterMixin, TransformerMixin

from scikit_learn_bench import default_params, profiler, CONST, display
from scikit_learn_bench import Timeout

def _bench_1_model(
        model_constructor: Callable,
        X: np.ndarray,
        y: np.ndarray,
        data_params: Dict[str, int],
        min_prof_time: float,
        max_prof_time: float,
        results: Dict[str, Union[Tuple[float, float], Tuple[float, float, float, float]]],
        profilerImpl: profiler.ProfilerStrategy
) -> None:
    model_name = model_constructor.__name__
    train_result = profilerImpl.get_default()
    infer_result = profilerImpl.get_default()

    try:
        kwargs, X, y = default_params.get_ml_default_params(model_name, X, y, data_params)
        model = model_constructor(**kwargs)
        model_display_name = model.__class__.__name__
    except Exception as e:
        if CONST.DISPLAY_WARNING:
            print(f"[Constructor Error] {model_name}: {e}")
        return

    try:
        if CONST.IS_MAX_PROF_TIME:
            train_result, model = Timeout.timeout_warp(profilerImpl.profile_training, timeout=max_prof_time)(model.fit, min_prof_time, X, y)
        else:
            train_result = profilerImpl.profile(model.fit, min_prof_time, X, y)
    except Exception as e:
        if CONST.DISPLAY_WARNING:
            print(f"[Training Error] {model_display_name}.fit: {e}")

    is_fitted = train_result != profilerImpl.get_default()
    if is_fitted:
        try:
            if CONST.IS_MAX_PROF_TIME:
                infer_result = Timeout.timeout_warp(profilerImpl.profile, timeout=max_prof_time)(model.predict, min_prof_time, X)
            else:
                infer_result = profilerImpl.profile(model.predict, min_prof_time, X)
        except Exception as e:
            if CONST.DISPLAY_WARNING:
                print(f"[Inference Error] {model_display_name}.predict: {e}")

    def comb(train_result, infer_result):
        if isinstance(train_result,tuple) and isinstance(infer_result, tuple):
            return train_result+infer_result
        else:
            return (train_result, infer_result)
    results[model_display_name] = comb(train_result, infer_result)

def get_db(ml_category: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for benchmarking.

    Parameters
    ----------
    ml_category : str
        Machine learning category: 'reg', 'cla', 'clu', or 'tra'.
    kwargs : dict
        Parameters passed to dataset generators.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features and labels.
    """
    if ml_category in {"reg", "tra"}:
        kwargs["n_targets"] = kwargs.pop("num_output")
        return make_regression(**kwargs)
    elif ml_category in {"cla", "clu"}:
        kwargs.update({
            "n_classes": kwargs.pop("num_output"),
            "n_redundant": 0,
            "n_repeated": 0,
            "n_clusters_per_class": 1
        })
        return make_classification(**kwargs)
    else:
        raise ValueError(f"Unknown ML category: {ml_category}")


def _is_transformer_only(constructor: Callable) -> bool:
    """Check if a model is a pure transformer (not regressor, classifier, or clusterer)."""
    is_transformer = issubclass(constructor, TransformerMixin)
    non_transformer_mixins = (RegressorMixin, ClusterMixin, ClassifierMixin)
    return is_transformer and not any(issubclass(constructor, mix) for mix in non_transformer_mixins)


def get_constructors(ml_category: str) -> List[Callable]:
    """
    Retrieve appropriate model constructors for the specified ML category.

    Parameters
    ----------
    ml_category : str
        One of: 'reg', 'cla', 'clu', 'tra'.

    Returns
    -------
    List[Callable]
        List of estimator constructors.
    """
    if ml_category == "reg":
        mixin = RegressorMixin
    elif ml_category == "cla":
        mixin = ClassifierMixin
    elif ml_category == "clu":
        mixin = ClusterMixin
    elif ml_category == "tra":
        mixin = TransformerMixin
    else:
        raise ValueError(f"Unsupported ML category: {ml_category}")

    constructors = []
    for name, constructor in all_estimators():
        if issubclass(constructor, mixin):
            if mixin == TransformerMixin and not _is_transformer_only(constructor):
                continue
            constructors.append((name, constructor))

    return constructors


def bench(num_samples: int = 1000,
          num_features: int = 100,
          num_output: int = 2,
          min_prof_time: float = 0.1,
          max_prof_time: float = 60.,
          ml_type: str = "cla",
          profiler_type: str = "time",
          table_print: bool = True,
          table_print_sort_crit: int = 1,
          line_profiler_path: str = "."
          ) -> Dict[str, Tuple]:
    """
    Benchmark scikit-learn models for training/inference time (and memory optionally).

    Parameters
    ----------
    num_samples : int
        Number of training samples.
    num_features : int
        Number of features.
    num_output : int
        Output dimensionality: targets, classes, or clusters.
    min_prof_time : float
        Max time allowed for training each model (in seconds).
    ml_type : str
        Type of models: 'cla' (default), 'reg', 'clu', 'tra', 'all'.
    profiler_type : str, optional
        Type of profiler: "time" (default), "timememory", "timeline".
    table_print : bool, optional
        If True, prints formatted table of results.
    table_print_sort_crit : int, optional
        Sort by: 0 - name, index in the `score.values() - 1`
        1 - training time, 2 - inference time or training memory (of TimeMemoryProfiler), 3 - ...

    Returns
    -------
    Dict[str, Tuple]
        Benchmark results keyed by model name.
    """
    if not CONST.DISPLAY_WARNING:
        import warnings
        for warning_type in [DeprecationWarning, FutureWarning, UserWarning, RuntimeWarning, ConvergenceWarning]:
            warnings.filterwarnings("ignore", category=warning_type)

    assert ml_type in {"reg", "cla", "clu", "tra"}, "Invalid ML category"

    model_constructors = get_constructors(ml_type)

    data_params = {
        "n_samples": num_samples,
        "n_features": num_features,
        "num_output": num_output,
        "n_informative": num_features,
        "random_state": 0
    }
    X_train, y_train = get_db(ml_type, **data_params)

    if profiler_type=="time":
        prof= profiler.TimeProfiler()
    elif profiler_type=="timememory":
        prof= profiler.TimeMemoryProfiler()
    elif profiler_type=="timeline":
        prof= profiler.TimeLineProfiler(line_profiler_path)
    else:
        raise ValueError("Error profiler type not understood: ", profiler_type)

    results = {}
    for _, constructor in model_constructors:
        _bench_1_model(constructor, X_train, y_train, data_params, min_prof_time, max_prof_time, results, prof)

    if table_print:
        display.print_table(results, table_print_sort_crit)

    return results
