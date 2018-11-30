import logging
import pandas as pd

logger = logging.getLogger(__name__)


def numeric_impute(train_df, test_df, numeric_feats):
    """
    This function will take the median of each of the columns and use that to impute missing data.

    :param train_df:
    :param test_df:
    :param numeric_feats:
    :return:
    """
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(numeric_feats, list)

    # takes the median per numeric feature and convert to dictionary
    impute_vals = train_df[numeric_feats].median(axis=0).to_dict()

    # imputes missing value
    train_filled_df = train_df.fillna(impute_vals)
    test_filled_df = test_df.fillna(impute_vals)

    return train_filled_df, test_filled_df


def main(train_df, test_df, numeric_feats, text_feats=None, cat_feats=None):
    """
    This function is responsible for imputing data of all kinds according to some simple rules.

    Read the README to see the simple rules.

    :param train_df:
    :param test_df:
    :param numeric_feats:
    :param text_feats:
    :param cat_feats:
    :return:
    """
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(numeric_feats, list)

    logger.info("Percentage of missing values in numeric features: \n{}".format(pd.concat([train_df[numeric_feats],
                                                                                           test_df[numeric_feats]],
                                                                                          ignore_index=True)
                                                                                .isnull().mean()))

    # imputes numeric values
    train_filled_df, test_filled_df = numeric_impute(train_df, test_df, numeric_feats)

    # impute text features with empty string
    if text_feats is None:
        pass
    else:
        assert isinstance(text_feats, list)

        train_filled_df = train_filled_df.fillna({col: '' for col in text_feats})
        test_filled_df = test_filled_df.fillna({col: '' for col in text_feats})

    # impute cat features with -1
    if cat_feats is None:
        pass
    else:
        assert isinstance(cat_feats, list)

        logger.info("Percentage of missing values in categorical features: \n{}".format(pd.concat([train_df[cat_feats],
                                                                                                   test_df[cat_feats]],
                                                                                                  ignore_index=True)
                                                                                        .isnull().mean()))

        train_filled_df = train_filled_df.fillna({col: -1 for col in cat_feats})
        test_filled_df = test_filled_df.fillna({col: -1 for col in cat_feats})

    logger.info("Number of samples with missing values in training dataset: {}".format(train_filled_df.isnull()
                                                                                       .any(axis=1).sum()))
    logger.info("Number of samples with missing values in testing dataset: {}".format(test_filled_df.isnull()
                                                                                      .any(axis=1).sum()))

    return train_filled_df, test_filled_df
