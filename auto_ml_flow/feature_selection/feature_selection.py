import pandas as pd
import numpy as np
from scipy.stats import f as f_dist


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

    We assume the group variance are all the "same".

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
