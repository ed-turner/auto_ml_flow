import logging

import pandas as pd
import numpy as np
from scipy.stats import f as f_dist

logger = logging.getLogger(__name__)


def two_tail_hypothesis_test(df, binary_col, val_col, confidence_level=1.96):
    """
    This will detect the difference of the groups of binary values and the value column.

    :param df:
    :param binary_col:
    :param val_col:
    :param confidence_level:
    :return:
    """

    assert isinstance(df, pd.DataFrame)
    assert df[binary_col].nunique() == 2

    mean_dict = df.groupby(binary_col)[val_col].mean().to_dict()

    var_dict = df.groupby(binary_col)[val_col].var().to_dict()

    count_dict = df[binary_col].value_counts().to_dict()

    cat_vals = list(mean_dict.keys())

    mu1 = mean_dict.get(cat_vals[0])
    sigma1 = var_dict.get(cat_vals[0])
    n1 = count_dict.get(cat_vals[0])

    mu2 = mean_dict.get(cat_vals[1])
    sigma2 = var_dict.get(cat_vals[1])
    n2 = count_dict.get(cat_vals[1])

    top = mu1 - mu2
    bot = ((sigma1 / n1) + (sigma2 / n2)) ** 0.5

    return abs(top / bot) > confidence_level


def one_way_anova_test(df, cat_col, val_col, alpha=0.05):
    """
    This will detect the difference of the groups of values and the value column.
    The return value is whether it is safe to assume there are group differences.

    We assume the group variance are all the "same", and that the val_col is normal.

    For more information, consider http://www-hsc.usc.edu/~eckel/biostat2/notes/notes7.pdf

    :param df:
    :param cat_col:
    :param val_col:
    :param alpha:
    :return:
    """

    assert isinstance(df, pd.DataFrame)
    assert df[cat_col].nunique() > 2

    mu = df[val_col].mean()

    mean_dict = df.groupby(cat_col)[val_col].mean().to_dict()

    count_dict = df[cat_col].value_counts().to_dict()

    cat_vals = list(mean_dict.keys())

    # number of groups
    k = len(cat_vals)
    n = df.shape[0]

    # deviation within the groups
    ssw = 0.0

    # deviation between the groups
    ssb = 0.0

    # uses the sample data calculate the ssw and ssb parameter
    for cat_val in cat_vals:

        mu1 = mean_dict.get(cat_val)
        n1 = count_dict.get(cat_val)

        indices = df[cat_col] == cat_val

        ssw += (df.loc[indices, cat_col] - mu1).sum() ** 2.0
        ssb += n1 * ((mu1 - mu) ** 2.0)

    # calculate the mean squared deviation
    msb = ssb / (k - 1)
    msw = ssw / (n - k)

    # calculate the f statistic
    f_stat = msb / msw

    # calculate the p-value
    p_value = f_dist.cdf(f_stat, k - 1, n - k)

    return p_value < alpha


def local_correlation(df, val_col, pred_col, n=50):
    """
    This will take the value column and calculate the correction to the prediction column in an interval.

    Then we will calculate the average correlation values.

    :param df:
    :param val_col:
    :param pred_col:
    :param n:
    :return:
    """

    assert isinstance(df, pd.DataFrame)
    assert isinstance(n, int)

    corr_vals = []

    min_val = df[val_col].min()
    max_val = df[val_col].max()

    dv = (max_val - min_val) / (n - 1)

    for i in range(n-1):

        x_lower = min_val + dv*i
        x_upper = min_val + dv*(i+1)

        indices = (df[val_col] <= x_upper) & (x_lower <= df[val_col])

        local_val_series = df.loc[indices, val_col].reset_index(drop=True)
        local_pred_series = df.loc[indices, pred_col].reset_index(drop=True)

        corr_vals.append(local_val_series.corr(local_pred_series))

    avg_corr = np.tanh(np.mean(np.arctanh(np.abs(corr_vals))))

    return avg_corr


def main(train_df, test_df, numeric_feats, pred_col, cat_feats=None, corr_thrshld=0.3):
    """
    This mean function will loop through each numeric feature and selection which feature is statistically relevant to
    the prediction column.

    We will also go through each categorical feature and determine which feature has the most differences per group.

    We assume the prediction column is normally distributed for the entire process.

    :param train_df:
    :param test_df:
    :param numeric_feats:
    :param pred_col:
    :param cat_feats:
    :param corr_thrshld:
    :return:
    """

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    drop_cols = []

    # per feature, we calculate the local correlation
    for feat in numeric_feats:
        avg_local_corr = local_correlation(train_df, feat, pred_col)

        # if it is below our threshold, we drop
        if avg_local_corr < corr_thrshld:
            drop_cols.append(feat)

    if cat_feats is None:
        pass
    else:
        assert isinstance(cat_feats, list)

        for feat in cat_feats:
            # if we only have one group in our cat_feat, we drop
            if train_df[feat].nunique() == 1:
                drop_cols.append(feat)
            elif train_df[feat].nunique() == 2:

                # if the two tail test fails, we skip
                if two_tail_hypothesis_test(train_df, feat, pred_col):
                    pass
                # else we add the feat to the drop list
                else:
                    drop_cols.append(feat)
            else:
                # if our group are the same, we drop

                if one_way_anova_test(train_df, feat, pred_col):
                    pass
                else:
                    drop_cols.append(feat)

    logger.info("Features to drop from training and testing datasets: {}\n".format(drop_cols))

    return train_df.drop(drop_cols, axis=1), test_df.drop(drop_cols, axis=1)