import click
import pandas as pd
from pathlib import Path
import pickle
import os
import sys

from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, fbeta_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.mean_std_cv_scores import mean_std_cross_val_scores
from utils.models import get_models

    
@click.command()
@click.option('--train-data', required=True, help='Path to train data CSV')
@click.option('--target-col', required=True, help='Name of the target column')
@click.option('--preprocessor-path', required=True, help='Path to preprocessor')
@click.option('--pos-label', default='Heart Disease', help='Positive class label for fbeta_score')
@click.option('--beta', default=2.0, help='Beta parameter for fbeta_score')
@click.option('--random-state', default=123, help='Random state for classifiers')
@click.option('--results', required=True, help='File path to save results table, include name of the CSV file e.g., results/CV_scores_default_parameters.csv')

def main(train_data, target_col, preprocessor_path, pos_label, beta, random_state, results):
    """
    Evaluate default models using cross-validation and save results.
    Parameters
    ----------
    train_data : str
        Path to training data CSV file.
    target_col : str
        Name of the target column in the dataset.
    preprocessor_path : str
        Path to the preprocessor pickle file.
    pos_label : str
        Positive class label for fbeta_score.
    beta : float
        Beta parameter for fbeta_score.
    random_state : int
        Random state for classifiers.
    results : str
        File path to save results table.
    """

    df = pd.read_csv(train_data)

    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    models = get_models(random_state=random_state)
    scorer = make_scorer(fbeta_score, pos_label=pos_label, beta=beta)

    results_dict = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        results_dict[name] = mean_std_cross_val_scores(
            pipe, X_train, y_train, cv=5, return_train_score=True, scoring=scorer
        )
    results_df = pd.DataFrame(results_dict).T

    results_path = Path(results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_path, index=True)


if __name__ == "__main__":
    main()
