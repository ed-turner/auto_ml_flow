import logging

import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def kfold_prediction(est, x_train, x_test, y_train, num_folds=5):
    """
    This is used to help create unbiased predictions.

    :param x_train:
    :param x_test:
    :param y_train:
    :param num_folds:
    :return:
    """

    train_kfold_pred = np.zeros(shape=(y_train.shape[0],))
    test_kfold_pred_list = []

    val_score_list = []

    kf = KFold(n_splits=num_folds)

    for train_index, test_index in kf.split(x_train):

        x, y = x_train[train_index, :], y_train[train_index]

        x_val, y_val = x_train[test_index, :], y_train[test_index]

        est.fit(x, y)

        train_kfold_pred[test_index] = est.predict(x_val)

        val_score_list.append(est.score(x_val, y_val))

        test_kfold_pred_list.append(est.predict(x_test))

    test_kfold_pred = np.mean(np.column_stack(test_kfold_pred_list), axis=0)

    return train_kfold_pred, test_kfold_pred


def main(x_train, x_test, y_train, model_dict):
    """
    This is used create lower-varianced solutions

    :param x_train:
    :param x_test:
    :param y_train:
    :param model_dict:
    :return:
    """

    train_preds = []
    test_preds = []

    if model_dict is None:
        model_dict = {'linear': LinearRegression()}

    for model_name in model_dict.keys():
        if model_name == 'ensemble':
            pass
        else:
            est = model_dict.get(model_name)

            train_kfold_pred, test_kfold_pred = kfold_prediction(est, x_train, x_test, y_train)

            train_preds.append(train_kfold_pred)
            test_preds.append(test_kfold_pred)

    train_preds = np.column_stack(train_preds)
    test_preds = np.column_stack(test_preds)

    est = model_dict.get("ensemble", "simple")

    if est == "simple":
        if test_preds.shape[1] == 1:
            return test_preds
        else:
            return np.mean(test_preds, axis=0)

    else:

        est.fit(train_preds, y_train)

        return est.predict(test_preds)
