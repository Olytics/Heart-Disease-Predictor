from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def get_models(random_state=123):
    
    return {
        "dummy clf": DummyClassifier(strategy='most_frequent'),
        "decision tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "RBF SVM": SVC(random_state=random_state)
    }