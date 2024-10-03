import warnings
import multiprocessing
from typing import Tuple, List
import pandas as pd
import numpy as np
import logging
import sys
import time
import re

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from codecarbon import EmissionsTracker
from data import upload_to_db

tracker = EmissionsTracker(
    project_name="skt_codecarbon",
    measure_power_secs=10,
    save_to_file=False
)

RB_COLUMNS = ['RB_800', 'RB_1800', 'RB_2100', 'RB_2600_10', 'RB_2600_20']
TARGET_COLUMN = 'RBused'
TRAIN_RATIO = 0.8
N_JOBS = max(1, multiprocessing.cpu_count() // 4)

warnings.filterwarnings("ignore")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r'[^\w\s]', '_', col) for col in df.columns]
    df.columns = ['feature_'+col if col[0].isdigit() else col for col in df.columns]
    df.columns = [col.strip('_') for col in df.columns]
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    for col in ['Holiday', 'Vendor']:
        df[col] = df[col].astype('category').cat.codes

    targets = [TARGET_COLUMN] + RB_COLUMNS
    for target in targets:
        df[f'{target}_lag_1h'] = df.groupby(['enb_id', 'pci'])[target].shift(1)
        df[f'{target}_lag_24h'] = df.groupby(['enb_id', 'pci'])[target].shift(24)
        df[f'{target}_lag_168h'] = df.groupby(['enb_id', 'pci'])[target].shift(168)

    for target in targets:
        rolling = df.groupby(['enb_id', 'pci'])[target].rolling(window=24)
        df[f'{target}_rolling_mean_24h'] = rolling.mean().reset_index(level=[0,1], drop=True)
        df[f'{target}_rolling_std_24h'] = rolling.std().reset_index(level=[0,1], drop=True)

    df['User_RB_ratio'] = df['Usertotal'] / df['RBtotal']
    df['timestamp_int'] = df['timestamp'].astype(int) // 10**9

    df = clean_column_names(df)

    return df

def train_lightgbm_model(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[lgb.Booster, List[str]]:
    feature_columns = [col for col in train.columns if col not in [TARGET_COLUMN] + RB_COLUMNS and train[col].dtype in ['int64', 'float64', 'bool']]
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': N_JOBS
    }

    train_data = lgb.Dataset(train[feature_columns], label=train[TARGET_COLUMN])
    valid_data = lgb.Dataset(test[feature_columns], label=test[TARGET_COLUMN])

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    feature_importance = model.feature_importance()
    top_features = [feature_columns[i] for i in np.argsort(feature_importance)[-20:]]

    return model, top_features

def forecast(train: pd.DataFrame, test: pd.DataFrame, model: lgb.Booster, top_features: List[str]) -> pd.DataFrame:
    forecast = pd.DataFrame(index=test.index)
    forecast[TARGET_COLUMN] = model.predict(test[top_features])
    
    for col in RB_COLUMNS:
        forecast[col] = test[col]  # Assuming we're using actual RB values for simplicity

    return forecast

def evaluate(actual: pd.DataFrame, forecast: pd.DataFrame) -> Tuple[float, float, float]:
    mae = mean_absolute_error(actual[TARGET_COLUMN], forecast[TARGET_COLUMN])
    mse = mean_squared_error(actual[TARGET_COLUMN], forecast[TARGET_COLUMN])
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_features(df)
    df.dropna(inplace=True)

    train = df.iloc[:int(TRAIN_RATIO*len(df))]
    test = df.iloc[int(TRAIN_RATIO*len(df)):]

    return train, test

def run(df: pd.DataFrame, district_gu: str, district_dong: str, algorithm: str):
    logging.info("Starting LightGBM model...")
    logging.debug(f"Data sample:\n{df.head()}")
    start_time = time.time()
    tracker.start()
    try:
        train, test = preprocess_data(df)

        model, top_features = train_lightgbm_model(train, test)
        forecast_results = forecast(train, test, model, top_features)

        mae, mse, rmse = evaluate(test, forecast_results)
        logging.info(f"Evaluation metrics - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    finally:
        carbon_emissions: float = tracker.stop() * 1000

    logging.info("LightGBM model completed.")

    end_time = time.time()
    total_duration = end_time - start_time

    metric = {
        'district_gu': district_gu,
        'district_dong': district_dong,
        'algorithm': algorithm,
        'duration': total_duration,
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'carbon_emissions': carbon_emissions
    }

    logging.info(f"Total execution time: {total_duration:.2f} seconds")
    logging.info(f"Carbon emissions: {metric['carbon_emissions']:.3f} gCO2")

    logging.info(f"Results:\n{forecast_results.head()}")

    upload_to_db(metric, forecast_results, district_gu, district_dong, algorithm)

    return test, forecast_results
