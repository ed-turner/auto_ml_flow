import logging

import numpy as np
import pandas as pd

import numdifftools.nd_algopy as nda

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def recall_loss(y_true, y_pred, tol):
    """
    This is the smooth version of recall

    :param y_true:
    :param y_pred:
    :param tol:
    :return:
    """
    tp = np.dot(y_true, y_pred)
    possible_p = np.sum(y_true)

    return tp / (possible_p + tol)


def precision_loss(y_true, y_pred, tol):
    """
    This is the smooth version of precision

    :param y_true:
    :param y_pred:
    :param tol:
    :return:
    """
    tp = np.dot(y_true, y_pred)
    predict_p = np.sum(y_pred)

    return tp / (predict_p + tol)


def f1_loss(y_true, y_pred, tol=1e-14):
    """
    This is the smooth version of f1 loss

    :param y_true:
    :param y_pred:
    :param tol:
    :return:
    """
    precision = precision_loss(y_true, y_pred, tol)
    recall = recall_loss(y_true, y_pred, tol)

    f1 = (2 * recall * precision) / (recall + precision + tol)

    return f1


def f1_obej(y_true, y_pred):
    """
    This is the objective function to return the gradient and hessian for the LGB and XGB models
    :param y_true:
    :param y_pred:
    :return:
    """
    def funct(x):
        return f1_loss(y_true, x)

    grad = nda.Gradient(funct)(y_pred)
    hess = nda.Hessian(funct)(y_pred)

    return grad, hess


def lgbm_f1_score(y_true, y_pred):
    """
    This is a scorer function for f1

    :param y_true:
    :param y_pred:
    :return:
    """
    return 'fl_score', f1_score(y_true, (y_pred > 0.5).astype(int)), True


def stratified_k_fold_preds(est, train_data, train_vals, test_data, is_lgbm=True):
    """
    This is to help handle class imbalance

    :param est:
    :param train_data:
    :param train_vals:
    :param test_data:
    :param is_lgbm:
    :return:
    """
    train_preds = np.zeros((train_vals.shape[0],))
    test_preds = []

    avg_fscores = []

    skf = StratifiedKFold(n_splits=5)

    i = 1

    for train_index, test_index in skf.split(train_data, train_vals):
        logger.info("Iteration {}".format(i))
        i += 1
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_vals[train_index], train_vals[test_index]

        if is_lgbm:
            est.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20,
                    eval_metric=lgbm_f1_score)
        else:
            est.fit(X_train, y_train)

        train_pred_fold = est.predict_proba(X_test)[:, 1]

        avg_fscores.append(f1_score(y_test, (train_pred_fold > 0.5).astype(int)))
        train_preds[test_index] = train_pred_fold
        test_preds.append(est.predict_proba(test_data)[:, 1])

        logger.info("Average F1-Score: {}".format(np.mean(avg_fscores)))

    return train_preds, np.column_stack(test_preds).mean(axis=1)


def get_thsh(stack_preds, train_vals):
    """
    This is to help get the threshold to decided whether to indentify a label or not

    :param stack_preds:
    :param train_vals:
    :return:
    """
    skf = StratifiedKFold(n_splits=5)

    thsh_f1_scores = []

    for train_index, test_index in skf.split(stack_preds, train_vals):

        train_preds, val_preds = stack_preds[train_index], stack_preds[test_index]

        y_train, y_val = train_vals[train_index], train_vals[test_index]

        thsh_f1_scores += [[thsh,
                            f1_score(y_train, (train_preds > thsh).astype(int)),
                            f1_score(y_val, (val_preds > thsh).astype(int))] for thsh in np.arange(0.501, 0.01)]

    return thsh_f1_scores


def opt_f1_test_preds(train_stack_pred, train_vals, test_stack_pred):
    """

    :param train_stack_pred:
    :param train_vals:
    :param test_stack_pred:
    :return:
    """
    thsh_f1_scores = get_thsh(train_stack_pred, train_vals)

    thsh_df = pd.DataFrame(data=thsh_f1_scores, columns=['thsh', 'train_f1_score', 'val_f1_score'])

    thsh_df = thsh_df.groupby('thsh').mean().reset_index(drop=True).sort_values(by=['train_f1_score', 'val_f1_score'],
                                                                                ascending=False)

    thsh = thsh_df.loc[0, 'thsh']

    return (test_stack_pred > thsh).astype(int)
