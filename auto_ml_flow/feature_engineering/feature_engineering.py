# data management
import numpy as np
from scipy.sparse import hstack

# for textual features
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def poly_features():
    raise NotImplementedError


def categorical_interactions():
    raise NotImplementedError


def engineer_text_features(train_df, test_df, textual_features_dict):
    """
    We have a textual features dictionary to allow the user to specify per textual feature whether we want to use
    the CountVectorizer or the TfidfVectorizer with their desired parameters

    :param train_df:
    :param test_df:
    :param textual_features_dict:
    :return:
    """

    train_list = []
    test_list = []

    for feat in textual_features_dict.keys():

        model_type = textual_features_dict.get('model_type', 'tfidf')
        model_params = textual_features_dict.get('model_params', {})

        if model_type == 'count':
            model = CountVectorizer(**model_params)
        elif model_type == 'tfidf':
            model = TfidfVectorizer(**model_params)
        else:
            raise ValueError

        model.fit(train_df[feat])

        train_list.append(model.transform(train_df[feat]))
        test_list.append(model.transform(test_df[feat]))

    return hstack(train_list), hstack(test_list)


def engineer_dummy_features(train_df, test_df, cat_feats):
    """

    We loop through each categorical feature, remove one group, and use binary identification to one-hot encode features

    :param train_df:
    :param test_df:
    :param cat_feats:
    :return:
    """

    train_list = []
    test_list = []

    for feat in cat_feats:

        # we get the unique, sort
        cat_vals = np.sort(train_df[feat].unique())

        # we ignore values of -1, as that is the imputed values
        if cat_vals[0] == -1:
            cat_vals = cat_vals[1:]
        # if we do not have missing values, we just drop the last cat val
        else:
            cat_vals = cat_vals[:-1]

        for val in cat_vals:
            train_list.append((train_df[feat] == val).astype(int))
            test_list.append((test_df[feat] == val).astype(int))

    return np.column_stack(train_list), np.column_stack(test_list)


def main(train_df, test_df, numeric_feats, cat_feats=None, textual_features_dict=None):
    """

    :param train_df:
    :param test_df:
    :param numeric_feats:
    :param cat_feats:
    :param textual_features_dict:
    :return:
    """

    x_train = train_df[numeric_feats]
    x_test = test_df[numeric_feats]

    if cat_feats is None:
        pass
    else:
        cat_train, cat_test = engineer_dummy_features(train_df, test_df, cat_feats)

        x_train = np.column_stack((x_train, cat_train))
        x_test = np.column_stack((x_test, cat_test))

    if textual_features_dict is None:
        pass
    else:
        text_train, text_test = engineer_text_features(train_df, test_df, textual_features_dict)

        x_train = hstack((x_train, text_train))
        x_test = hstack((x_test, text_test))

    return x_train, x_test