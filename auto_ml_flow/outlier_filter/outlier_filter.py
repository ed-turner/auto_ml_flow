import logging

import numpy as np

logger = logging.getLogger(__name__)


def determine_outlier(x, is_weak=True):
    """
    This will determine what observations are outliers.

    :param x:
    :param is_weak:
    :return:
    """

    # finds the percentiles of x
    x_25 = np.percentile(x, 25)
    x_med = np.median(x)
    x_75 = np.percentile(x, 75)

    # finds the interquartile range of x
    x_qr = x_75 - x_25

    # if we are looking for weak outliers, we define a larger region to observe
    if is_weak:
        alpha = 3.0
    else:
        alpha = 1.5

    x_upper = x_med + alpha*x_qr
    x_lower = x_med - alpha*x_qr

    indices = (x_lower < x) & (x < x_upper)

    return indices


def main(df, features):
    """
    We loop through each feature and determine which observations is an outlier for each feature. Then we remove it.

    :param df:
    :param features:
    :return:
    """

    df_filtered = df.copy()

    logger.info("Number of samples before: {}".format(df_filtered.shape[0]))

    for feat in features:
        df_filtered = df_filtered.loc[determine_outlier(df_filtered[feat]), :].reset_index(drop=True)

    logger.info("Number of samples after: {}".format(df_filtered.shape[0]))

    return df_filtered
