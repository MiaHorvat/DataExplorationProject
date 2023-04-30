from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np

def load_data(housing_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads the dataset from a given file path, applies some data cleaning and feature engineering,
    and returns the selected features and target variable

    """
    df = pd.read_csv("housing.csv")
    df = df[df['median_house_value'] <= 500000]
    df = df[df['housing_median_age'] <= 45]
    df.drop(columns=['ocean_proximity'], inplace=True)
    df['distance_to_beach'] = ((df['longitude'] - (-118.4902))**2 + (df['latitude'] - 34.0195)**2)**0.5
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('median_house_value', axis=1), df['median_house_value'], test_size=test_size, random_state=random_state)
    selected_features = ['distance_to_beach', 'median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'latitude', 'longitude']
    return df[selected_features], df['median_house_value']

def combine_features(train_data: pd.DataFrame, test_data: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function combines features in the train and test dataframes by adding two new columns:
    'avg_rooms_per_household' and 'avg_occupancy_per_household'.
    It calculates the average number of rooms per household and the average occupancy per household.

    """
    train_data['avg_rooms_per_household'] = train_data['total_rooms'] / train_data['households']
    test_data['avg_rooms_per_household'] = test_data['total_rooms'] / test_data['households']
    train_data['avg_occupancy_per_household'] = train_data['population'] / train_data['households']
    test_data['avg_occupancy_per_household'] = test_data['population'] / test_data['households']
    return train_data, test_data
   

def encode_features(train_data: pd.DataFrame, test_data: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function encodes categorical features in the train and test dataframes by creating dummy variables
    using pd.get_dummies method.
    """
    train_data_encoded = pd.get_dummies(train_data, drop_first=True)
    test_data_encoded = pd.get_dummies(test_data, drop_first=True)
    return train_data_encoded, test_data_encoded

def scale_features(train_data_encoded: pd.DataFrame, test_data_encoded: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale the encoded training and testing data using StandardScaler
    """
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_encoded)
    test_data_scaled = scaler.transform(test_data_encoded)
    return train_data_scaled, test_data_scaled

def train_model(train_data_scaled: pd.DataFrame, train_target: pd.Series, parameters: dict) -> GridSearchCV:
    """
    Train an XGBoostRegressor model using the scaled training data and perform hyperparameter tuning using GridSearchCV
    """
    model = XGBRegressor()
    grid_search = GridSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=2)
    with mlflow.start_run():
        grid_search.fit(train_data_scaled, train_target)
        mlflow.log_params(grid_search.best_params_)
    return grid_search

def evaluate_model(model: xgb.XGBRegressor, test_data_scaled: np.ndarray, test_target: pd.Series) -> None:
    """
    Evaluates the trained XGBRegressor model on the test data and prints the model's performance metrics
    (R2 score, mean absolute error, mean squared error, and root mean squared error) on the test set.
    Additionally, logs these metrics to an MLflow experiment.
    """
    predictions = model.predict(test_data_scaled)
    score = model.score(test_data_scaled, test_target)
    mae = mean_absolute_error(test_target, predictions)
    mse = mean_squared_error(test_target, predictions)
    rmse = mean_squared_error(test_target, predictions, squared=False)
    print("Bestimmtheitsmaß R2 des Modells: {:.2f}".format(score))
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("Mean Squared Error (MSE): {:.2f}".format(mse))
    print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
    mlflow.log_metrics({'R2': score, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})

def run_experiment(file_path: str, parameters: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    This function runs an experiment to train and evaluate an XGBoost model on the provided dataset.
    """
    train_data, train_target = load_data(file_path)
    test_size=0.2
    random_state=42
    train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=test_size, random_state=random_state)
    train_data, test_data = combine_features(train_data, test_data)
    train_data_encoded, test_data_encoded = encode_features(train_data, test_data)
    train_data_scaled, test_data_scaled = scale_features(train_data_encoded, test_data_encoded)
    model = train_model(train_data_scaled, train_target, parameters)
    evaluate_model(model, test_data_scaled, test_target)
    return train_data, test_data, train_target, test_target

# Define dictionary containing hyperparameter search space
parameters = {'n_estimators': [50, 100, 150],
'max_depth': [3, 5, 7],
'learning_rate': [0.1, 0.01, 0.001]}

run_experiment('housing.csv', parameters)
