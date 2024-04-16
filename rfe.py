import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from math import sqrt
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import pathlib
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
import sklearn
import xgboost
from xgboost import XGBRegressor
import glob
import os
import sqlalchemy
from datetime import datetime
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from joblib import dump, load

print('Starting data loading process')

###all_states.csv is all gauntlet features except lat, lon, and sqft for all US states within US Structures (https://gis-fema.hub.arcgis.com/pages/usa-structures)
df = 'all_states.csv'

#Ensure that df read in correctly and you have all necessary features
print(df.columns)

#Data should be 31 million rows
print("Length of dataframe is: ", len(df))
print('\n')

#split into train / test
print('Starting engineering / splitting data')

def filter_and_evaluate(df):
    """
    filter_and_evaluate Filters a DataFrame to remove buildings with a height less than 2 meters and visualizes the height distribution.

    Parameters:
    - df (DataFrame): Input DataFrame containing height information.

    Returns:
    - df_filtered (DataFrame): Filtered DataFrame with buildings taller than 2 meters. This removal of buildings <2m is consistent with previous research
    
    The function filters the input DataFrame to retain only the rows where the 'height' column is greater than 2 meters.
    It then computes the mean absolute error (MAE) and root mean squared error (RMSE) between the actual heights
    and the median height of the filtered DataFrame, providing insights into the height distribution.
    The 'median' column is added temporarily for evaluation purposes and dropped before returning the filtered DataFrame.
    """
    df_filtered = df[df['height'] > 2]

    # Generating a median column to understand height distribution
    df_filtered['median'] = df_filtered['height'].median()

    height = df_filtered['height'].to_numpy()
    median = df_filtered['median'].to_numpy()

    print('MAE:', mean_absolute_error(height, median))
    print('RMSE:', np.sqrt(mean_squared_error(height, median)))

    df_filtered.drop(columns=['median'], inplace=True)
    
    return df_filtered


def preprocess_data(filtered_df):
    """
    Preprocesses a DataFrame containing building height information for machine learning tasks.

    Parameters:
    - filtered_df (DataFrame): Input DataFrame containing filtered building data with height greater than 2 meters.

    Returns:
    - X_train_scaled (DataFrame): Scaled features for the training set.
    - X_test_scaled (DataFrame): Scaled features for the testing set.
    - y_train (Series): Target variable (building height) for the training set.
    - y_test (Series): Target variable (building height) for the testing set.
    - scaler (StandardScaler): Scaler object used for standardization.

    The function takes a DataFrame containing filtered building data, where buildings with a height less than
    2 meters have been removed. It splits the data into features (X) and target variable (y), then further splits
    it into training and testing sets using a 70-30 split, with a fixed random state for reproducibility.

    The features are then standardized using a StandardScaler, which is fitted on the training data and applied
    to both the training and testing sets. The scaler object is saved for future use.

    The function returns the scaled features for the training and testing sets, along with the target variables
    for both sets and the scaler object.
    """
    X = filtered_df.drop(columns=['height'])
    y = filtered_df['height']

    # Splitting into train / test split - please note the random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

    print('length of training data:', len(X_train))
    print('length of testing data:', len(X_test))
    print('\n')

    # Standardizing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Saving the scaler
    dump(scaler, 'my_standard_scaler.pkl')

    # Applying the same scaler to test data
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def XGB(X_train, X_test, y_train, y_test):
    """
    Trains an XGBoost regression model and evaluates its performance on training and testing sets to reduce dimensionality.

    Parameters:
    - X_train (DataFrame): Features of the training set.
    - X_test (DataFrame): Features of the testing set.
    - y_train (Series): Target variable (building height) for the training set.
    - y_test (Series): Target variable (building height) for the testing set.

    The function trains an XGBoost regression model on the provided training data and evaluates its performance
    on both the training and testing sets. It prints the R-squared (R^2) score, mean absolute error (MAE),
    and root mean squared error (RMSE) for both sets to assess the model's predictive accuracy.

    Additionally, the function performs recursive feature elimination to identify the most important features
    contributing to the model's performance. The selected features are printed for reference.

    Returns:
    - MAE
    - RMSE
    - R^2
    - Selected features

    The function does not return any value explicitly; it prints the evaluation metrics and feature selection results.
    """
    print('Starting the ml process now')

    # Train the XGBoost model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Training Evaluation
    print("XGBoost performance for training set")
    print("--------------------------------------")
    train_pred = model.predict(X_train)
    print('R^2:', r2_score(y_train, train_pred))
    print('MAE:', mean_absolute_error(y_train, train_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_train, train_pred)))
    print('\n')

    # Testing Evaluation
    test_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)

    print("XGBoost performance for testing set")
    print("--------------------------------------")
    print('R2 score is {}'.format(r2))
    print('MAE: {}'.format(mae))
    print('RMSE is {}'.format(rmse))
    print("\n")

    # Recursive feature elimination
    model_2 = RFE(estimator=XGBRegressor(), n_features_to_select=None)
    _ = model_2.fit(X_train, y_train)

    X_train_columns = X_train.loc[:, model_2.support_]
    columns = X_train_columns.columns

    print(len(X_train.columns))
    print(columns)

filtered_df = filter_and_evaluate(df)
X_train, X_test, y_train, y_test, scaler = preprocess_data(filtered_df)
XGB(X_train, X_test, y_train, y_test)
