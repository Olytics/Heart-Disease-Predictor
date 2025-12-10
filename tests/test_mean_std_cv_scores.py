import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, fbeta_score

from utils.mean_std_cv_scores import mean_std_cross_val_scores

def test_mean_std_cv_scores():
    
    X = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "chol": [180, 200, 190, 210, 220]
    })
    y = pd.Series(["No Heart Disease", "Heart Disease", "No Heart Disease", "Heart Disease", "No Heart Disease"])

    scoring = make_scorer(fbeta_score, beta=2, pos_label="Heart Disease")

    lr = make_pipeline(StandardScaler(), LogisticRegression())

    scores = mean_std_cross_val_scores(
        model=lr, 
        X_train=X,
        y_train=y,
        scoring=scoring,
        cv=3,
        return_train_score=True
    )


    assert isinstance(scores, pd.Series)
    assert "train_score" in scores.index
    assert "test_score" in scores.index


    assert isinstance(scores["train_score"], str)
    assert "(" in scores["train_score"]
    assert "+/-" in scores["train_score"]