import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .data import get_all_data
from .utils import generate_submission, EventTimer

def build_model(cfg, global_settings):
    mapping = {
        'RandomForestClassifier': RandomForestClassifier,
        'SVC': SVC,
        'XGBClassifier': XGBClassifier,
        'LGBMClassifier': LGBMClassifier,
    }

    clf = mapping[cfg.name](**cfg.args)

    param_grid = cfg.param_grid if cfg.param_grid is not None else {}

    return GridSearchCV(clf, param_grid,
                        scoring=make_scorer(matthews_corrcoef),
                        n_jobs=global_settings.n_jobs,
                        )


def train(cfg, datadir, model_path, prediction_path):
    ((train_users, train_features), train_labels), (test_users, test_features), user_features = get_all_data(datadir)

    # Handle data
    train_user_matrix = np.stack([user_features[user] for user in train_users])
    train_matrix = np.concatenate([train_user_matrix, train_features], axis=1)

    clf = build_model(cfg.model, cfg.global_settings)
    print(clf)

    with EventTimer('Fitting the model'):
        clf.fit(train_matrix, train_labels)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    # test on training set
    train_preds = clf.predict(train_matrix)
    print(f'> train_acc: {accuracy_score(train_labels, train_preds)}')
    print(f'> train_mcc: {matthews_corrcoef(train_labels, train_preds)}')

    # Handle test data
    test_user_matrix = np.stack([user_features[user] for user in test_users])
    test_matrix = np.concatenate([test_user_matrix, test_features], axis=1)

    test_preds = clf.predict(test_matrix)
    generate_submission(test_preds, prediction_path)
