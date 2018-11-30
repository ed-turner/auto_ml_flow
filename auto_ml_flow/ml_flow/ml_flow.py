import logging

import pandas as pd

from auto_ml_flow.imputation.imputation import main as impute
from auto_ml_flow.outlier_filter.outlier_filter import main as outlier_filter
from auto_ml_flow.feature_selection.feature_selection import main as select_feature
from auto_ml_flow.feature_engineering.feature_engineering import main as engineer_feature
from auto_ml_flow.prediction.prediction import main as predict

logger = logging.getLogger(__name__)


def exec(train_df, test_df, pred_col, numeric_feats=None, cat_feats=None, text_features=None, remove_outliers=False,
         select_features=False, engineer_features=None, model_dict=None):
    """

    :param train_df:
    :param test_df:
    :param pred_col:
    :param numeric_feats:
    :param cat_feats:
    :param text_features:
    :param remove_outliers:
    :param select_features:
    :param engineer_features:
    :param model_dict:
    :return:
    """

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    assert isinstance(pred_col, str)

    if numeric_feats is None:
        numeric_feats = list(set(train_df.columns) - set(pred_col))

    if train_df.isnull().any(axis=None) | test_df.isnull().any(axis=None):
        logger.info("We are imputing missing values")

        train_data, test_data = impute(train_df, test_df, numeric_feats, cat_feats=cat_feats, text_feats=text_features)

    else:
        train_data, test_data = train_df.copy(), test_df.copy()

    assert ~(train_data.isnull().any(axis=None)) & (test_data.isnull().any(axis=None))

    if remove_outliers:
        logger.info("We are removing any outliers from the training dataset using the numeric features and " +
                    "prediction column")
        train_data = outlier_filter(train_data, numeric_feats + [pred_col])
    else:
        pass

    # get the y-value
    y_train = train_data[pred_col]

    if select_features:
        logger.info("Performing feature selection.")

        try:
            train_data, test_data = select_feature(train_data, test_data, numeric_feats, pred_col, cat_feats)
        except Exception as e:
            logger.fatal(e, exc_info=True)
    else:
        pass

    if engineer_features is None:
        pass
    else:
        logger.info("Engineering features")

        if isinstance(engineer_features, dict):
            if text_features is None:
                textual_features_dict = None
            else:
                textual_features_dict = engineer_features.get('text', {feat: dict() for feat in text_features})

        else:
            if text_features is None:
                textual_features_dict = None
            else:
                textual_features_dict = {feat: dict() for feat in text_features}

        try:
            train_data, test_data = engineer_feature(train_data, test_data, numeric_feats, cat_feats,
                                                     textual_features_dict)
        except Exception as e:
            logger.fatal(e, exc_info=True)

    logger.info("Making predictions on the final testing dataset.")

    return predict(train_data, test_data, y_train, model_dict)