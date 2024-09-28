import pandas as pd
import numpy as np
import logging
import time
import io
import gc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from numba import cuda

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
SEQUENCE_LENGTH = 10
NUM_FEATURES = len(RB_COLUMNS)
FUTURE_STEPS = 1  # RBused
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 128

scaler = MinMaxScaler()

def prepare_data(df):
    data_dict = {}

    for enbid in df['enbid_pci'].unique():
        enbid_df = df[df['enbid_pci'] == enbid].copy()

        features = enbid_df[RB_COLUMNS]
        target = enbid_df[TARGET_COLUMN]

        scaled_features = scaler.fit_transform(features)

        scaled_df = pd.DataFrame(scaled_features, index=enbid_df.index, columns=RB_COLUMNS)
        scaled_df[TARGET_COLUMN] = target.values

        train_size = int(len(scaled_df) * TRAIN_RATIO)
        train_data, test_data = scaled_df[:train_size], scaled_df[train_size:]

        logging.debug(f"Enbid: {enbid} - Train Data Start: {train_data.index[0]}, Train Data End: {train_data.index[-1]}")
        logging.debug(f"Enbid: {enbid} - Test Data Start: {test_data.index[0]}, Test Data End: {test_data.index[-1]}")

        data_dict[enbid] = {
            'train': train_data,
            'test': test_data
        }

    return data_dict

def create_sequences(data, target_column, time_steps=10):
    X = []
    y = []
    timestamps = []

    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i+time_steps][RB_COLUMNS].values)
        y.append(data.iloc[i+time_steps][target_column])
        timestamps.append(data.index[i+time_steps])

    logging.debug(f"Sequences Timestamps - Start: {timestamps[0]}, End: {timestamps[-1]}")

    return np.array(X), np.array(y), np.array(timestamps)

def prepare_lstm_data(data_dict, time_steps=10):
    logging.info("Preparing LSTM data...")
    lstm_data_dict = {}

    for enbid, data in data_dict.items():
        train_data, test_data = data['train'], data['test']

        X_train, y_train, train_timestamps = create_sequences(train_data, TARGET_COLUMN, time_steps)

        X_test, y_test, test_timestamps = create_sequences(test_data, TARGET_COLUMN, time_steps)

        lstm_data_dict[enbid] = {
            'X_train': X_train,
            'y_train': y_train,
            'train_timestamps': train_timestamps,
            'X_test': X_test,
            'y_test': y_test,
            'test_timestamps': test_timestamps
        }

    return lstm_data_dict

def create_lstm_model():
    logging.info("Creating LSTM model...")
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(FUTURE_STEPS)
    ])
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse")
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()
    logging.info("\n\nModel Summary:\n" + summary_str)
    return model

def train_and_predict_lstm(lstm_data_dict):
    model_dict = {}
    model = create_lstm_model()
    i = 1

    mae_list, mse_list, rmse_list = [], [], []

    for enbid, data in lstm_data_dict.items():
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        logging.info(f"Processing PCI: {enbid} ({i}/{len(lstm_data_dict)})")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=0)
        logging.info(f"Model training finished - epochs: {EPOCHS}")
        forecast = model.predict(X_test, verbose=0)
        logging.info(f"Model prediction finished")
        mae, mse, rmse = evaluate(y_test, forecast)
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        logging.info(f"Evaluation metrics for PCI {enbid} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
        model_dict[enbid] = {
            'model': model,
            'forecast': forecast,
            'test_timestamps': data['test_timestamps']
        }
        i += 1
    return model_dict, mae_list, mse_list, rmse_list

def evaluate(actual, forecast):
    forecast = forecast.flatten()
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def create_forecast_dataframe(model_dict):
    forecast_df = pd.DataFrame()

    for enbid, data in model_dict.items():
        forecasts = data['forecast'].flatten()
        timestamps = data['test_timestamps']
        enbid_forecast_df = pd.DataFrame(forecasts, index=timestamps, columns=[enbid])
        forecast_df = pd.concat([forecast_df, enbid_forecast_df], axis=1)

    return forecast_df

def cleanup_gpu_memory():
    tf.keras.backend.clear_session()

    try:
        cuda.select_device(0)
        cuda.close()
    except:
        pass

    gc.collect()
    tf.compat.v1.get_default_graph().finalize()
    time.sleep(1)

def run(df, district_gu, district_dong, algorithm):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # GPU 메모리 점진적 할당 설정
            logging.info("Dynamic memory growth enabled for GPUs")
        except RuntimeError as e:
            logging.error(e)
    start_time = time.time()
    logging.info("Starting LSTM model...")
    logging.debug(f"Data sample:\n{df.head()}")
    tracker.start()
    try:
      data_dict = prepare_data(df)
      lstm_data_dict = prepare_lstm_data(data_dict)
      model_dict, mae_list, mse_list, rmse_list = train_and_predict_lstm(lstm_data_dict)
      forecast_df = create_forecast_dataframe(model_dict)
    finally:
        carbon_emissions: float = tracker.stop() * 1000

    logging.info("LSTM model completed.")

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

    logging.info(f"Total duration: {total_duration:.2f} seconds")
    logging.info(f"Average evaluation metrics - MAE: {metric['mae']:.2f}, MSE: {metric['mse']:.2f}, RMSE: {metric['rmse']:.2f}")
    logging.info(f"Carbon emissions: {metric['carbon_emissions']:.3f} gCO2")

    logging.info(f"Results:\n{forecast_df.head()}")

    upload_to_db(metric, forecast_df, district_gu, district_dong, algorithm)

    cleanup_gpu_memory()
    logging.info("GPU memory cleaned up.")

