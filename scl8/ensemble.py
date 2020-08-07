import pickle

import numpy as np

from .data import get_all_test_data
from .utils import generate_submission

def ensemble(cfg, datadir, prediction_path, mode='average'):
    (test_users, test_features), user_features = get_all_test_data(datadir)
    test_user_matrix = np.stack([user_features[user] for user in test_users])
    test_matrix = np.concatenate([test_user_matrix, test_features], axis=1)

    model_outputs = []
    weights = []

    for model_desc in cfg.models:
        with open(model_desc.model_path, 'rb') as f:
            model = pickle.load(f)

        print(f'> model loaded: "{str(model)[:30]}" ...')

        if mode == 'average':
            model_outputs.append(model.predict_proba(test_matrix)[:, 1])
        elif mode == 'voting':
            model_outputs.append(model.predict(test_matrix))

        weights.append(model_desc.weight)

    print(np.stack(model_outputs).shape)
    predictions = np.around(
        np.average(np.stack(model_outputs), axis=0, weights=weights)
    ).astype('int')

    generate_submission(predictions, prediction_path)
