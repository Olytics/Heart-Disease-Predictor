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
from utils.optimal_hyperparameters import tune_hyperparameters

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
    assert isinstance(result.best_score_, float)

def test_tune_hyperparameters_with_pos_label_0(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    result = tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=0, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)
    assert isinstance(result.cv_results_, dict)

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
    X_train = pd.DataFrame({'feature_0': [1, 2, 3, 5, 3, 5, 4, 3, 1, 3], 'feature_1': [7, 8, 9, 10, 11, 12, 11, 9, 6, 8]})
    y_train = pd.Series([0, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    preprocessor = StandardScaler()
    model = LogisticRegression(random_state=42)
    param_dist = {'logisticregression__C': [0.1, 1.0]}
    result = tune_hyperparameters(X_train, y_train, model, preprocessor, param_dist, pos_label=1, beta=2, seed=42)
    assert isinstance(result, RandomizedSearchCV)

# Abnormal, error or adversarial use cases
def test_tune_hyperparameters_invalid_pos_label(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    with pytest.raises(ValueError):
        tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=-1, beta=2, seed=42)

def test_tune_hyperparameters_empty_param_dist(sample_data, sample_preprocessor, sample_model):
    X_train, y_train = sample_data
    param_dist = {}
    with pytest.raises(ValueError):
        tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, param_dist, pos_label=1, beta=2, seed=42)

def test_tune_hyperparameters_invalid_beta(sample_data, sample_preprocessor, sample_model, sample_param_dist):
    X_train, y_train = sample_data
    with pytest.raises(ValueError):
        tune_hyperparameters(X_train, y_train, sample_model, sample_preprocessor, sample_param_dist, pos_label=1, beta=-1, seed=42)