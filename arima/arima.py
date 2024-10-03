import warnings
import multiprocessing
from typing import Tuple, List
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import sys
import itertools
import time

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

from sklearn.metrics import mean_absolute_error, mean_squared_error

from codecarbon import EmissionsTracker
from data import upload_to_db

tracker = EmissionsTracker(
    project_name="skt_codecarbon",
    measure_power_secs=10,
    save_to_file=False
)

RB_COLUMNS = ['RB_800', 'RB_1800', 'RB_2100', 'RB_2600_10', 'RB_2600_20']
TRAIN_RATIO = 0.8
N_JOBS = max(1, multiprocessing.cpu_count() // 4)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

def evaluate_model(params: Tuple[int, int, int], train: pd.Series, train_excluded: pd.DataFrame) -> Tuple[Tuple[int, int, int], float]:
    try:
        model = SARIMAX(train, exog=train_excluded, order=params)
        model_fit = model.fit(disp=False)
        return (params, round(model_fit.aic, 2))
    except:
        return (params, float('inf'))

def greedy_search(train: pd.Series, train_excluded: pd.DataFrame) -> List[Tuple[Tuple[int, int, int], float]]:
    p = range(0, 3)
    d = range(1, 2)
    q = range(0, 6)

    pdq = list(itertools.product(p, d, q))

    results = Parallel(n_jobs=N_JOBS)(delayed(evaluate_model)(i, train, train_excluded) for i in tqdm(pdq, file=sys.stdout, ncols=70, desc="Optimizing"))

    aic = [r[1] for r in results]
    optimal = [r for r in results if r[1] == min(aic)]

    return optimal

def fit_and_forecast(train: pd.DataFrame, test: pd.DataFrame, frequency: str, model: lgb.Booster, top_features: List[str]) -> Tuple[str, pd.Series]:
    feature_columns = [col for col in top_features if col != frequency]
    train_data = lgb.Dataset(train[feature_columns], label=train[frequency])
    
    params = model.params
    frequency_model = lgb.train(params, train_data, num_boost_round=model.best_iteration)
    
    forecast_result = frequency_model.predict(test[feature_columns])
    return (frequency, pd.Series(forecast_result, index=test.index))

def forecast(train: pd.DataFrame, test: pd.DataFrame, model: lgb.Booster, top_features: List[str]) -> pd.DataFrame:
    forecast = pd.DataFrame()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(fit_and_forecast)(train, test, frequency, model, top_features) 
        for frequency in tqdm(RB_COLUMNS, desc="Forecasting")
    )
    for frequency, result in results:
        forecast[frequency] = result
    forecast['RBused'] = forecast[RB_COLUMNS].sum(axis=1)
    return forecast

def evaluate(actual: pd.DataFrame, forecast: pd.Series) -> Tuple[float, float, float]:
    mae = mean_absolute_error(actual['RBused'], forecast)
    mse = mean_squared_error(actual['RBused'], forecast)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby('enbid_pci').resample('15min').mean().reset_index()
    grouped.set_index(['enbid_pci', 'timestamp'], inplace=True)
    grouped = grouped.unstack(level='enbid_pci').swaplevel(axis=1).sort_index(axis=1)
    grouped.fillna(0, inplace=True)

    train = grouped.iloc[:int(TRAIN_RATIO*len(grouped))]
    test = grouped.iloc[int(TRAIN_RATIO*len(grouped)):]

    return train, test

def run(df: pd.DataFrame, district_gu: str, district_dong: str, algorithm: str):
    logging.info("Starting ARIMA model...")
    logging.debug(f"Data sample:\n{df.head()}")
    start_time = time.time()
    tracker.start()
    try:
        train, test = preprocess_data(df)

        forecast_results = pd.DataFrame()
        total_pci = len(train.columns.levels[0])

        mae_list, mse_list, rmse_list = [], [], []

        for idx, pci in enumerate(train.columns.levels[0], 1):
            logging.info(f"Processing PCI: {pci} ({idx}/{total_pci})")

            optimal = greedy_search(train[pci]['RB_800'], train[pci][RB_COLUMNS].drop(columns=['RB_800']))
            forecast_results[pci] = forecast(train, test, pci, optimal[0])['RBused']

            logging.info(f"Optimal Parameters: {optimal[0][0]}")
            mae, mse, rmse = evaluate(test[pci], forecast_results[pci])
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            logging.info(f"Evaluation metrics - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    finally:
        carbon_emissions: float = tracker.stop() * 1000

    logging.info("ARIMA model completed.")

    end_time = time.time()
    total_duration = end_time - start_time

    metric = {
        'district_gu': district_gu,
        'district_dong': district_dong,
        'algorithm': algorithm,
        'duration': total_duration,
        'mae': float(np.mean(mae_list)),
        'mse': float(np.mean(mse_list)),
        'rmse': float(np.mean(rmse_list)),
        'carbon_emissions': carbon_emissions
    }

    logging.info(f"Total execution time: {total_duration:.2f} seconds")
    logging.info(f"Average evaluation metrics - MAE: {metric['mae']:.2f}, MSE: {metric['mse']:.2f}, RMSE: {metric['rmse']:.2f}")
    logging.info(f"Carbon emissions: {metric['carbon_emissions']:.3f} gCO2")

    logging.info(f"Results:\n{forecast_results.head()}")

    upload_to_db(metric, forecast_results, district_gu, district_dong, algorithm)

    return test, forecast_results