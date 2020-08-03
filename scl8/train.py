import numpy as np
import pickle
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .data import get_all_data
from .utils import seed_everything, generate_submission
from .utils import EventTimer

def build_model(cfg, grid_args):
    mapping = {
        'RandomForestClassifier': RandomForestClassifier,
        'SVC': SVC,
        'XGBClassifier': XGBClassifier,
        'LGBMClassifier': LGBMClassifier,
    }

    clf = mapping[cfg.name](**cfg.args)

    param_grid = dict(cfg.param_grid) if cfg.param_grid is not None else {}

    return GridSearchCV(clf, param_grid, **grid_args)


def train(cfg, datadir, model_path, prediction_path, random_state):
    seed_everything(random_state)

    ((train_users, train_features), train_labels), (test_users, test_features), user_features = get_all_data(datadir)

    # Handle data
    train_user_matrix = np.stack([user_features[user] for user in train_users])
    train_matrix = np.concatenate([train_user_matrix, train_features], axis=1)

    train_x, val_x, train_y, val_y = train_test_split(train_matrix, train_labels, test_size=0.2, random_state=random_state)

    clf = build_model(cfg.model, cfg.grid_args)
    print(clf)

    with EventTimer('Fitting the model'):
        clf.fit(train_x, train_y)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f'{"-"*12} (CV Results) {"-"*12}')
    pprint(clf.cv_results_)

    print(f'{"-"*12} (BEST PARAMS) {"-"*12}')
    print(clf.best_params_)

    # test on training set
    train_preds = clf.predict(train_x)
    print(f'> train_acc: {accuracy_score(train_y, train_preds)}')
    print(f'> train_mcc: {matthews_corrcoef(train_y, train_preds)}')

    # Validation
    val_preds = clf.predict(val_x)
    print(f'> val_acc: {accuracy_score(val_y, val_preds)}')
    print(f'> val_mcc: {matthews_corrcoef(val_y, val_preds)}')

    # Handle test data
    test_user_matrix = np.stack([user_features[user] for user in test_users])
    test_matrix = np.concatenate([test_user_matrix, test_features], axis=1)

    test_preds = clf.predict(test_matrix)
    generate_submission(test_preds, prediction_path)
