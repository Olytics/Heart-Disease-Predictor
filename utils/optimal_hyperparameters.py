from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline

def tune_hyperparameters(model, preprocessor, param_dist, pos_label, beta, seed):
    model = make_pipeline(preprocessor, model)
    search_model = RandomizedSearchCV(model, param_dist, return_train_score=True, random_state=seed,
                                    n_jobs=-1, scoring=make_scorer(fbeta_score, pos_label=pos_label, beta=beta))
    return search_model