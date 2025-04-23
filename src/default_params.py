import numpy as np
from typing import Any, Dict

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier, RandomForestRegressor, VotingRegressor, StackingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import (
    ClassifierChain, RegressorChain, MultiOutputClassifier, MultiOutputRegressor
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

def get_ml_args(ml_construct_name: str, X, y, data_params) -> dict[str, Any]:
    """
    Return default arguments for a given scikit-learn model constructor name.

    Parameters
    ----------
    ml_construct_name : str
        The name of the machine learning model class.
    X : array-like
        Input features (used only for context-specific defaults).
    y : array-like
        Target array to determine output dimensionality.

    Returns
    -------
    dict[str, Any]
        A dictionary of keyword arguments to initialize the model.
    """

    # Define a single base classifier and a list of classifiers
    base_classifier = DecisionTreeClassifier(random_state=42)
    classifier_ensemble = [
        ("dt", base_classifier),
        ("lr", LogisticRegression(random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
        ("knn", KNeighborsClassifier()),
        ("gnb", GaussianNB())
    ]

    # Define a single base regressor and a list of regressors
    base_regressor = LinearRegression()
    regressor_ensemble = [
        ("lr", base_regressor),
        ("dt", DecisionTreeRegressor(random_state=42)),
        ("svr", SVR()),
        ("knn", KNeighborsRegressor()),
        ("rf", RandomForestRegressor(random_state=42))
    ]

    auto_base_estim = LogisticRegression() if np.issubdtype(y.dtype, np.integer) else LinearRegression()
    transformers= [
        ("std_scaler", StandardScaler()),
        ("minmax_scaler", MinMaxScaler()),
        ("normalizer", Normalizer())]
    n_components = data_params["num_output"]

    # Default parameters by model name
    default_params = {
        # Classification wrappers
        "ClassifierChain": {"base_estimator": base_classifier},
        "OneVsOneClassifier": {"estimator": base_classifier},
        "OneVsRestClassifier": {"estimator": base_classifier},
        "OutputCodeClassifier": {"estimator": base_classifier},
        "StackingClassifier": {"estimators": classifier_ensemble},
        "VotingClassifier": {"estimators": classifier_ensemble},
        "MultiOutputClassifier": {"estimator": base_classifier},

        # Regression wrappers
        "RegressorChain": {"base_estimator": base_regressor},
        "StackingRegressor": {"estimators": regressor_ensemble},
        "VotingRegressor": {"estimators": regressor_ensemble},
        "MultiOutputRegressor": {"estimator": base_regressor},

        # Projection-based models
        "PLSCanonical": {"n_components": n_components},
        "CCA": {"n_components": n_components},
        "PLSSVD": {"n_components": n_components},

        # Feature selectors
        "SelectFromModel": {"estimator": base_classifier},
        "SequentialFeatureSelector": {"estimator": auto_base_estim},
        "TSNE": {"perplexity": min(10, max(1, len(y) // 2))},
        "SelectKBest": {"k": min(10, max(1, X.shape[1] // 2))},
        "RFE": {"estimator": auto_base_estim},
        "FeatureUnion": {'transformer_list':transformers},
        "ColumnTransformer": {"transformers":transformers}
    }

    return default_params.get(ml_construct_name, {})

def data_processing_for_ml_algo(ml_construct_name: str, X, y) -> tuple:
    from sklearn.preprocessing import FunctionTransformer

    transformer = FunctionTransformer(lambda X: X.clip(min=0), validate=False)
    if ml_construct_name in {"ComplementNB", "MultinomialNB", "CategoricalNB", "MiniBatchNMF", "NMF"}:
        X_clipped = transformer.fit_transform(X)
        return X_clipped, y
    elif ml_construct_name=="MultiOutputClassifier":
        y_2d = y.reshape(-1, 1)
        return X, y_2d
    elif ml_construct_name in {"PoissonRegressor", "GammaRegressor"}:
        y_pos = np.abs(y) + 1e-3
        return X, y_pos
    elif ml_construct_name == "ClassifierChain":
        multi_label = np.vstack([(y % 2 == 0), (y % 3 == 0)]).T.astype(int)
        return X, multi_label
    else:
        return X, y

def get_ml_default_params(ml_construct_name: str, X, y, data_params: Dict[str, int]) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    kwargs=get_ml_args(ml_construct_name, X, y, data_params)
    X2, y2 = data_processing_for_ml_algo(ml_construct_name, X, y)
    return kwargs, X2, y2