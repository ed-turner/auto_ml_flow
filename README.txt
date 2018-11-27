This is a package that will automate the entire machine learning process, given certain parameters are passed.

The imputation module is responsible for imputing missing values in the dataset.  For cases of numeric data,
we will impute using the median, as it will more robust to the outliers in the dataset.  For cases of ordinal and
categorical data, we will just impute with -1 to assign missing values a new category.  For textual data, we will just
impute with an empty string.

The feature_selection module is responsible selecting features based on the statistical analysis of the data.  For
numeric data, we will, after standardizing the data and after setting a step-size, take the mean absolute
"local correlation" to determine whether a feature is dependent on the prediction variable.  For values too low (i.e
values below a set threshold), we will remove those features.  For ordinal/categorical features, we will perform
hypothesis test to determine if these features can be used to detect responsible differences in the data.

The outlier_filter module is responsible for noting outliers in the training dataset, and using an identification scheme
to detect outliers in the test dataset. If the flag is set, you can detect outliers in the prediction
per ordinal/categorical value.

The feature_engineering module is going to take a numeric data, if the flag is set, and create polynomial features of
the numeric data.  For ordinal and categorical data, it will, if the flag is set, create interaction features between
different ordinal and categorical data.  By default, it will take all ordinal and categorical data and dummy encode it.
For textual data, it will create TF-IDF features or Count Vectors of choice, by default. If a flag is set, it will use a
stemmer of choice.  However, this should be installed separately.

The prediction module is responsible using the final dataset, and the models specified in a yaml file with their
parameters, to create predictions.  For multiple models passed, the default ensemble model is a simple average.
A different ensemble model can be used.  Also, it can be used to use k-fold prediction.

The main function will run in the following order: imputation, feature selection, outlier filtration,
feature engineering and prediction.  It is possible to override feature selection, outlier filtration,
feature engineering by leaving those parameters as None.  The output will be the prediction of the test dataset.

The training dataset and testing dataset must be a pandas dictionary.