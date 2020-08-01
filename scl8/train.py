import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .data import get_all_data
from .utils import generate_submission

def train(datadir, output_file):
    ((train_users, train_features), train_labels), (test_users, test_features), user_features = get_all_data(datadir)

    train_user_matrix = np.stack([user_features[user] for user in train_users])
    train_matrix = np.concatenate([train_user_matrix, train_features], axis=1)

    train_x, val_x, train_y, val_y = train_test_split(train_matrix, train_labels)

    clf = SVC(kernel='linear')
    clf.fit(train_x, train_y)

    train_preds = clf.predict(train_x)
    print(f'> train_acc: {accuracy_score(train_y, train_preds)}')
    print(f'> train_mcc: {matthews_corrcoef(train_y, train_preds)}')

    val_preds = clf.predict(val_x)
    print(f'> val_acc: {accuracy_score(val_y, val_preds)}')
    print(f'> val_mcc: {matthews_corrcoef(val_y, val_preds)}')

    test_user_matrix = np.stack([user_features[user] for user in test_users])
    test_matrix = np.concatenate([test_user_matrix, test_features], axis=1)

    test_preds = clf.predict(test_matrix)
    generate_submission(test_preds, output_file)
