from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline

def tune_hyperparameters(X_train, y_train, model, preprocessor, param_dist, pos_label, beta, seed):
    model = make_pipeline(preprocessor, model)
    search_model = RandomizedSearchCV(model, param_dist, return_train_score=True, random_state=seed,
                                    n_jobs=-1, scoring=make_scorer(fbeta_score, pos_label=pos_label, beta=beta))
    return search_model.fit(X_train, y_train)

def get_best_model(model_summary: dict):
    results_dict = dict()
    best_score = 0
    best_model = None
    for model_name, summary in model_summary.items():
        results_dict[model_name] = [summary[1], summary[2]]
        print(f"The best F2 score for {model_name} is {summary[1]} with parameters {summary[2]}")
        best_score = max(best_score, summary[1])
        if best_score == summary[1]:
            best_model = model_name
        else:
            continue

    return model_summary[best_model][0].best_estimator_, results_dict