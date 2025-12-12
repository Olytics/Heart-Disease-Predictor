import pytest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.optimal_hyperparameters import tune_hyperparameters, get_best_model

# Fixtures for common test data
@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_train = pd.Series(y)
    return X_train, y_train

@pytest.fixture
def sample_preprocessor():
    return StandardScaler()

@pytest.fixture
def sample_model():
    return LogisticRegression(random_state=42)

@pytest.fixture
def sample_param_dist():
    return {'logisticregression__C': [0.1, 1.0, 10.0]}

# Tests for tune_hyperparameters function

# Simple expected use cases
def test_tune_hyperparameters_basic(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    result = tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=1, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)
    assert hasattr(result, 'best_estimator_')
    assert hasattr(result, 'best_score_')

def test_tune_hyperparameters_with_different_beta(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    result = tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=1, beta=1, seed=42)
    assert isinstance(result, RandomizedSearchCV)
    assert result.scoring == make_scorer(fbeta_score, pos_label=1, beta=1)

def test_tune_hyperparameters_with_pos_label_0(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    result = tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=0, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)
    assert result.scoring == make_scorer(fbeta_score, pos_label=0, beta=2)

# Edge cases
def test_tune_hyperparameters_single_param(sample_data, sample_preprocessor, sample_model):
    X_train, y_train = sample_data
    param_dist = {'logisticregression__C': [1.0]}
    result = tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, param_dist, pos_label=1, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)
    assert result.best_params_['logisticregression__C'] == 1.0

def test_tune_hyperparameters_large_param_space(sample_data, sample_preprocessor, sample_model):
    X_train, y_train = sample_data
    param_dist = {'logisticregression__C': np.logspace(-4, 4, 20)}
    result = tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, param_dist, pos_label=1, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)
    assert 'logisticregression__C' in result.best_params_

def test_tune_hyperparameters_small_dataset():
    X_train = pd.DataFrame({'feature_0': [1, 2, 3], 'feature_1': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    preprocessor = StandardScaler()
    model = LogisticRegression(random_state=42)
    param_dist = {'logisticregression__C': [0.1, 1.0]}
    result = tune_hyperparameters(X_train, y_train, model, preprocessor, param_dist, pos_label=1, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)

# Abnormal, error or adversarial use cases
def test_tune_hyperparameters_invalid_pos_label(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    with pytest.raises(ValueError):
        tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=2, beta=2, seed=42)

def test_tune_hyperparameters_empty_param_dist(sample_data, sample_preprocessor, sample_model):
    X_train, y_train = sample_data
    param_dist = {}
    with pytest.raises(ValueError):
        tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, param_dist, pos_label=1, beta=2, seed=42)

def test_tune_hyperparameters_invalid_beta(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    with pytest.raises(ValueError):
        tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=1, beta=-1, seed=42)

# Tests for get_best_model function

# Simple expected use cases
def test_get_best_model_single_model():
    model_summary = {
        'model1': (RandomizedSearchCV(LogisticRegression(), {}), 0.85, {'C': 1.0})
    }
    best_model, results_dict = get_best_model(model_summary)
    assert isinstance(best_model, LogisticRegression)
    assert results_dict == {'model1': [0.85, {'C': 1.0}]}

def test_get_best_model_multiple_models():
    model1 = RandomizedSearchCV(LogisticRegression(), {})
    model1.best_estimator_ = LogisticRegression(C=1.0)
    model2 = RandomizedSearchCV(LogisticRegression(), {})
    model2.best_estimator_ = LogisticRegression(C=0.1)
    model_summary = {
        'model1': (model1, 0.85, {'C': 1.0}),
        'model2': (model2, 0.90, {'C': 0.1})
    }
    best_model, results_dict = get_best_model(model_summary)
    assert best_model == model2.best_estimator_
    assert results_dict == {'model1': [0.85, {'C': 1.0}], 'model2': [0.90, {'C': 0.1}]}

def test_get_best_model_tie_scores():
    model1 = RandomizedSearchCV(LogisticRegression(), {})
    model1.best_estimator_ = LogisticRegression(C=1.0)
    model2 = RandomizedSearchCV(LogisticRegression(), {})
    model2.best_estimator_ = LogisticRegression(C=0.1)
    model_summary = {
        'model1': (model1, 0.85, {'C': 1.0}),
        'model2': (model2, 0.85, {'C': 0.1})
    }
    best_model, results_dict = get_best_model(model_summary)
    # Should return the first one encountered with max score
    assert best_model == model1.best_estimator_
    assert results_dict == {'model1': [0.85, {'C': 1.0}], 'model2': [0.85, {'C': 0.1}]}

# Edge cases
def test_get_best_model_empty_dict():
    model_summary = {}
    with pytest.raises(KeyError):
        get_best_model(model_summary)

def test_get_best_model_single_model_zero_score():
    model_summary = {
        'model1': (RandomizedSearchCV(LogisticRegression(), {}), 0.0, {'C': 1.0})
    }
    best_model, results_dict = get_best_model(model_summary)
    assert isinstance(best_model, LogisticRegression)
    assert results_dict == {'model1': [0.0, {'C': 1.0}]}

def test_get_best_model_negative_scores():
    model1 = RandomizedSearchCV(LogisticRegression(), {})
    model1.best_estimator_ = LogisticRegression(C=1.0)
    model2 = RandomizedSearchCV(LogisticRegression(), {})
    model2.best_estimator_ = LogisticRegression(C=0.1)
    model_summary = {
        'model1': (model1, -0.1, {'C': 1.0}),
        'model2': (model2, -0.05, {'C': 0.1})
    }
    best_model, results_dict = get_best_model(model_summary)
    assert best_model == model2.best_estimator_
    assert results_dict == {'model1': [-0.1, {'C': 1.0}], 'model2': [-0.05, {'C': 0.1}]}

# Abnormal, error or adversarial use cases
def test_get_best_model_invalid_tuple_length():
    model_summary = {
        'model1': (RandomizedSearchCV(LogisticRegression(), {}), 0.85)  # Missing params
    }
    with pytest.raises(IndexError):
        get_best_model(model_summary)

def test_get_best_model_non_numeric_score():
    model_summary = {
        'model1': (RandomizedSearchCV(LogisticRegression(), {}), 'high', {'C': 1.0})
    }
    with pytest.raises(TypeError):
        get_best_model(model_summary)

def test_get_best_model_none_values():
    model_summary = {
        'model1': (None, 0.85, {'C': 1.0})
    }
    with pytest.raises(AttributeError):
        get_best_model(model_summary)