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

df = df_us.csv

print('length of United States training dataframe is:', len(df))


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

filtered_df = filter_and_evaluate(df)


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

X_train, X_test, y_train, y_test, scaler = preprocess_data(filtered_df)

def fine_tune(space, X_train, y_train, X_test, y_test):
    """
    Conducts hyperparameter tuning and cross-validation for an XGBoost regression model,
    and evaluates the performance on a given test dataset.

    This function initializes an XGBoost regressor with hyperparameters specified in `space`,
    performs k-fold cross-validation to evaluate the model, and then refits the model on
    the entire training dataset before evaluating it on the test set. The function logs the
    cross-validation and test set performance metrics (RMSE, MAE, R^2) and prints the duration
    of the tuning process.

    Parameters:
    - space (dict): A dictionary containing the hyperparameters for the XGBoost model. Expected
      keys are 'n_estimators', 'max_depth', 'gamma', 'reg_alpha', 'reg_lambda', 'colsample_bytree',
      'min_child_weight', and 'learning_rate'.
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target variable.
    - X_test (DataFrame): Test data features.
    - y_test (Series): Test data target variable.

    Returns:
    - dict: A dictionary with the keys 'loss' (mean cross-validation MSE), 'status' (optimization
      status, typically STATUS_OK), 'model' (trained XGBoost model), 'test_rmse' (root mean squared
      error on the test set), 'test_mae' (mean absolute error on the test set), and 'test_r2' (R^2
      score on the test set).

    Note:
    - This function prints the mean RMSE from cross-validation, evaluation metrics from the test
      set, and the duration of the model training and evaluation process.
    """
    
    model = XGBRegressor(n_estimators=int(space['n_estimators']),
                         max_depth=int(space['max_depth']),
                         gamma=space['gamma'],
                         reg_alpha=space['reg_alpha'],
                         reg_lambda=space['reg_lambda'],
                         colsample_bytree=space['colsample_bytree'],
                         min_child_weight=space['min_child_weight'],
                         learning_rate=space['learning_rate'],
                         random_state=13)
    
    # Perform k-fold cross-validation (e.g., k=5)
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in kf.split(X_train):
        train_X_fold, val_X_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        train_y_fold, val_y_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
        
        model.fit(train_X_fold, train_y_fold)
        val_y_pred = model.predict(val_X_fold)
        mse = mean_squared_error(val_y_fold, val_y_pred)
        cv_scores.append(mse)
    
    # Calculate the mean RMSE (neg_mean_squared_error gives negative values)
    mse_mean = np.mean(cv_scores)
    print("Mean Cross-Validation RMSE:", np.sqrt(mse_mean))
    
    # Refit the model on the entire training data
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    test_y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_y_pred))
    test_mae = mean_absolute_error(y_test, test_y_pred)
    test_r2 = r2_score(y_test, test_y_pred)

    print("Test set evaluation:")
    print("RMSE:", test_rmse)
    print("MAE:", test_mae)
    print("R^2:", test_r2)

    
    return {
        'loss': mse_mean,
        'status': STATUS_OK,
        'model': model,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'max_depth': hp.quniform('max_depth', 2, 10, 1),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 5),
    'reg_lambda': hp.uniform('reg_lambda', 0, 5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
}

trials = Trials()
best = fmin(fn=lambda space: fine_tune(space, X_train, y_train, X_test, y_test),
            space=space,
            algo=tpe.suggest,
            max_evals=1,  # Increase max_evals for more iterations
            trials=trials)

best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'gamma': best['gamma'],
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda'],
    'colsample_bytree': best['colsample_bytree'],
    'min_child_weight': int(best['min_child_weight']),
    'learning_rate': best['learning_rate'],
}

print('\nBest hyperparameters:', best)

def ml(X_train, y_train, X_test, y_test, best_params):
    """
    Executes both linear regression and XGBoost regression models on the provided training and test datasets.
    This function is the fifth step in a series of functions designed for data preprocessing and modeling,
    following data cleaning, feature engineering, median calculation, and train-test-validation splitting.
    It assesses the performance of both models on the datasets and prints the evaluation metrics.

    Parameters:
    - X_train (DataFrame): The input features for the training set.
    - y_train (Series): The target variable for the training set.
    - X_test (DataFrame): The input features for the test set.
    - y_test (Series): The target variable for the test set.
    - best_params (dict): A dictionary containing the optimized hyperparameters for the XGBoost model.
      Expected keys are 'n_estimators', 'max_depth', 'gamma', 'reg_alpha', 'reg_lambda', 'colsample_bytree',
      'min_child_weight', and 'learning_rate'.

    Steps Performed:
    1. Linear Regression:
       - Fit the model on the training data.
       - Evaluate and print the RMSE, MAE, and R^2 metrics for both the training and test datasets.
    2. XGBoost Regression:
       - Fit the model with the provided `best_params` on the training data.
       - Evaluate and print the RMSE, MAE, and R^2 metrics for both the training and test datasets.

    Output:
    The function prints out the performance metrics for both models on both the training and testing sets,
    allowing for a comparison of model efficacy and transferability.

    Note:
    This function is primarily used for performance evaluation and does not return any values. Predictions are
    not transferred back into the original DataFrame within this function, but it includes comments suggesting
    that this could be a step to consider for error analysis.
    """
    print('Starting ml')
    # 1. Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    
    # Linear Regression evaluation for training set
    y_train_predict = lin_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    mae = mean_absolute_error(y_train, y_train_predict)
    r2 = r2_score(y_train, y_train_predict)

    print("The linear regression performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('MAE: {}'.format(mae))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    # Linear Regression evaluation for testing set
    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    mae = mean_absolute_error(y_test, y_test_predict)
    r2 = r2_score(y_test, y_test_predict)

    print("The linear regression performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('MAE: {}'.format(mae))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    # 2. XGBoost
    reg = XGBRegressor(n_estimators=int(best_params['n_estimators']),
                       max_depth=int(best_params['max_depth']),
                       gamma=best_params['gamma'],
                       reg_alpha=best_params['reg_alpha'],
                       reg_lambda=best_params['reg_lambda'],
                       colsample_bytree=best_params['colsample_bytree'],
                       min_child_weight=int(best_params['min_child_weight']),
                       learning_rate=best_params['learning_rate'],
                       random_state=13)
    # XGBoost evaluation for training set

    reg.fit(X_train, y_train)
    train_pred = reg.predict(X_train)
    print("The XGBoost performance for training set")
    print("--------------------------------------")
    print('RMSE:',np.sqrt(mean_squared_error(y_train, train_pred)))
    print('MAE:',mean_absolute_error(y_train, train_pred))
    print('R^2:',r2_score(y_train, train_pred))
    print('\n')
    
    # XGBoost evaluation for testing set
    target_test_pred = reg.predict(X_test)
    print("The XGBoost performance for testing set")
    print("--------------------------------------")
    print('RMSE:',np.sqrt(mean_squared_error(y_test, target_test_pred)))
    print('MAE:',mean_absolute_error(y_test, target_test_pred))
    print('R^2:',r2_score(y_test, target_test_pred))
    print('\n')

ml(X_train, y_train, X_test, y_test, best_params)


def cross_val(X_train, y_train):
    """
    Performs cross-validation to evaluate the performance of an XGBoost regression model using the training data.
    This function is positioned as the sixth step in a pipeline, following the instantiation of best hyperparameters.

    Steps:
    1. The function assumes that hyperparameters are defined in a dictionary named `best_params` globally.
    2. An XGBoost regressor model is instantiated with these parameters.
    3. The model is fit on the training data.
    4. The function sets up a 5-fold cross-validation scheme with shuffling.
    5. It calculates and stores the cross-validation scores (negative mean squared error) for the model.

    Parameters:
    - X_train (DataFrame): The training dataset features.
    - y_train (Series): The training dataset target variable.

    Returns:
    - scores (array): An array of the cross-validation scores for each fold, using negative mean squared error as the metric.

    Notes:
    - This function prints the mean square root of the negative mean squared errors from the cross-validation,
      which represents the average model performance across the folds.
    - Ensure that `best_params` dictionary is correctly defined in the scope where this function is called, as it uses
      these parameters to configure the XGBoost model.
    """
    print('Staring cross_val now')
    
    # 2. Instantiating a model with best_params from above
    xgb_cv=XGBRegressor(n_estimators=int(best_params['n_estimators']),
                       max_depth=int(best_params['max_depth']),
                       gamma=best_params['gamma'],
                       reg_alpha=best_params['reg_alpha'],
                       reg_lambda=best_params['reg_lambda'],
                       colsample_bytree=best_params['colsample_bytree'],
                       min_child_weight=int(best_params['min_child_weight']),
                       learning_rate=best_params['learning_rate'],
                       random_state=13)
    
    # 3. Fitting XGBoost on the training data
    xgb_cv.fit(X_train, y_train)

    # 4. determining number of splits on the data
    kfold = KFold(n_splits=5,
                  shuffle=True,
                  random_state=13)
    
    # 5. Running cross validation on the training data
    scores = cross_val_score(xgb_cv,
                             X_train,
                             y_train,
                             cv=kfold,
                             scoring='neg_mean_squared_error')
    

    mse = -np.mean(scores)
    print('Mean cross validation score: ', np.sqrt(mse))
    return scores

cross_val(X_train, y_train)





