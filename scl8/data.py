import calendar
import math
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

ONEHOT_FEATURES = [
    'country_code',
    'weekday',
]

USER_ONEHOT_FEATURES = [
    'domain',
]


def date_to_weekday(date: str):
    '''
    Args:
        date: a string, in format 'yyyy-mm-dd'
    Returns:
        an integer in [0, 6], 0 is Monday
    '''
    return calendar.weekday(*(map(int, date.split(' ')[0].split('-'))))


def onehot_encode(data):
    '''
    Args:
        data: should be 1d ndarray
    Returns:
        ndarray: shape(n_samples, ndims)
    '''
    onehot_encoder = OneHotEncoder(sparse=False)
    features = onehot_encoder.fit_transform(data.reshape(-1, 1))
    return features


def transform_feature_df(df, onehot_features=ONEHOT_FEATURES):
    users = df['user_id']
    df.drop(columns=['user_id'], inplace=True)

    # the index is row_id
    df.drop(columns=['row_id'], inplace=True)

    # max numerical data is 808 in train, 812 in test
    df['last_open_day'].replace({'Never open': 1000}, inplace=True)

    # max numerical data is 18141 in train, 18165 in test
    df['last_login_day'].replace({'Never login': 20000}, inplace=True)

    # max numerical data is 1445 in train, 1462 in test
    df['last_checkout_day'].replace({'Never checkout': 1500}, inplace=True)

    # add weekday to our data
    df.insert(len(df.columns), 'weekday', [date_to_weekday(date) for date in df['grass_date']])

    # drop grass date
    df.drop(columns=['grass_date'], inplace=True)

    features = []
    for col in df.columns:
        if col in onehot_features:
            features.append(onehot_encode(df[col].to_numpy()))
        else:
            features.append(df[col].to_numpy().reshape(-1, 1))

    feature_matrix = np.concatenate(features, axis=1)

    return users, feature_matrix


def transform_user_df(df, onehot_features=['attr_1', 'attr_2', 'attr_3', 'domain']):
    num_users = max(df['user_id']) + 1

    users = df['user_id']
    df.drop(columns=['user_id'], inplace=True)

    # fill attr_1 to be -1
    df['attr_1'].fillna(-1, inplace=True)

    # fill attr_2 to be -1
    df['attr_2'].fillna(-1, inplace=True)

    # fill age with mean
    df['age'].fillna(df['age'].mean(), inplace=True)

    unique_domains = set(df['domain'])
    domain_mapping = {name: i for i, name in enumerate(unique_domains)}

    df['domain'].replace(domain_mapping)

    features = []
    for col in df.columns:
        if col in onehot_features:
            features.append(onehot_encode(df[col].to_numpy()))
        else:
            features.append(df[col].to_numpy().reshape(-1, 1))

    feature_matrix = np.concatenate(features, axis=1)

    user_dict = {}
    for user_id, feature in zip(users, feature_matrix):
        user_dict[user_id] = feature

    return user_dict

def get_train_data(data_path: str) -> Tuple[np.ndarray, ...]:
    df = pd.read_csv(data_path)

    feature_df = df.drop(columns=['open_flag'])
    labels = df['open_flag'].to_numpy(dtype=int)

    return transform_feature_df(feature_df), labels


def get_test_data(data_path: str) -> np.ndarray:
    df = pd.read_csv(data_path)
    return transform_feature_df(df)


def get_user_data(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    return transform_user_df(df)


def get_all_data(datadir: str) -> Tuple[Any, ...]:
    train_features, train_labels = get_train_data(os.path.join(datadir, 'train.csv'))
    test_features = get_test_data(os.path.join(datadir, 'test.csv'))
    user_features = get_user_data(os.path.join(datadir, 'users.csv'))
    return (train_features, train_labels), test_features, user_features

def get_all_test_data(datadir: str) -> Tuple[Any, ...]:
    test_features = get_test_data(os.path.join(datadir, 'test.csv'))
    user_features = get_user_data(os.path.join(datadir, 'users.csv'))
    return test_features, user_features
