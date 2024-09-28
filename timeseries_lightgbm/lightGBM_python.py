import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import re
import matplotlib.pyplot as plt
import random

def clean_column_names(df):
    df.columns = [re.sub(r'[^\w\s]', '_', col) for col in df.columns]
    df.columns = ['feature_'+col if col[0].isdigit() else col for col in df.columns]
    df.columns = [col.strip('_') for col in df.columns]
    return df

def prepare_features(df):
    df = df.copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter_of_day'] = df['hour'] * 4 + df['minute'] // 15  # 0-95 for each 15-min interval in a day

    # Cyclical encoding for time variables
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    for col in ['Holiday', 'Vendor']:
        df[col] = df[col].astype('category').cat.codes

    targets = ['RBused', 'RB_800', 'RB_1800', 'RB_2100', 'RB_2600_10', 'RB_2600_20']

    # Lag features (past 1 hour, 3 hours, 1 day)
    lags = [4, 12, 96]  # 4 * 15min = 1h, 12 * 15min = 3h, 96 * 15min = 1 day
    for target in targets:
        for lag in lags:
            df[f'{target}_lag_{lag}'] = df.groupby(['enb_id', 'pci'])[target].shift(lag)

    # Rolling statistics (past 1 hour, 3 hours, 1 day)
    windows = [4, 12, 96]
    for target in targets:
        for window in windows:
            rolling = df.groupby(['enb_id', 'pci'])[target].rolling(window=window)
            df[f'{target}_rolling_mean_{window}'] = rolling.mean().reset_index(level=[0,1], drop=True)
            df[f'{target}_rolling_std_{window}'] = rolling.std().reset_index(level=[0,1], drop=True)

    df['User_RB_ratio'] = df['Usertotal'] / df['RBtotal']

    # Keep timestamp as integer for potential use
    df['timestamp_int'] = df['timestamp'].astype(int) // 10**9

    df = clean_column_names(df)

    return df

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return rmse, mae, r2

def train_lightgbm_model(X, y, tscv):
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
        'n_jobs': -1
    }

    train_data = lgb.Dataset(X, label=y)
    initial_model = lgb.train(params, train_data, num_boost_round=100)
    feature_importance = initial_model.feature_importance()
    top_features = X.columns[np.argsort(feature_importance)[-20:]].tolist()
    X = X[top_features]

    models = []
    scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        early_stopping_callback = lgb.early_stopping(stopping_rounds=50, verbose=False)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[early_stopping_callback]
        )

        y_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, y_pred))

        models.append(model)
        scores.append(score)

    best_model = models[np.argmin(scores)]
    return best_model, np.mean(scores), top_features

def plot_actual_vs_predicted(model, X, y, target):
    y_pred = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {target}')
    plt.show()

def predict_for_enbid_pci(model, df, enbid_pci, target, feature_columns):
    specific_data = df[df['enbid_pci'] == enbid_pci]

    if specific_data.empty:
        print(f"No data found for enbid_pci {enbid_pci}")
        return None, None

    X = specific_data[feature_columns]
    predictions = model.predict(X)

    return specific_data['timestamp'], predictions

def plot_actual_vs_predicted_timeseries(df, model, enbid_pci, target, feature_columns):
    timestamps, predictions = predict_for_enbid_pci(model, df, enbid_pci, target, feature_columns)

    if timestamps is None or predictions is None:
        return

    actual_values = df[df['enbid_pci'] == enbid_pci][target]

    plt.figure(figsize=(20, 5))
    plt.plot(timestamps, actual_values, label='Actual', alpha=0.7)
    plt.plot(timestamps, predictions, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title(f'Actual vs Predicted {target} for enbid_pci {enbid_pci}')
    plt.xlabel('Timestamp')
    plt.ylabel(target)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 메인 실행 코드
file_path = '/content/drive/MyDrive/Colab_Notebooks/Data2/ELG_Busan_PoC_per_CA_site_0226_0519.csv'
df = pd.read_csv(file_path)
df = prepare_features(df)

df = df.dropna()

targets = ['RBused', 'RB_800', 'RB_1800', 'RB_2100', 'RB_2600_10', 'RB_2600_20']

feature_columns = [col for col in df.columns if col not in targets and df[col].dtype in ['int64', 'float64', 'bool']]

tscv = TimeSeriesSplit(n_splits=5)

sample_size = int(len(df) * 0.3)
df_sampled = df.sample(n=sample_size, random_state=42)

models = {}
for target in targets:
    print(f"Training model for {target}...")
    X = df_sampled[feature_columns]
    y = df_sampled[target]
    model, score, top_features = train_lightgbm_model(X, y, tscv)
    models[target] = (model, top_features)
    print(f"{target} model average RMSE: {score}")

    # Model evaluation
    rmse, mae, r2 = evaluate_model(model, X[top_features], y)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    # Visualize actual vs predicted
    plot_actual_vs_predicted(model, X[top_features], y, target)

print("Training and evaluation complete!")


# RB_800에 대한 시계열 예측 시각화
target = 'RB_800'
random_enbid_pci = df['enbid_pci'].sample(n=1).iloc[0]
model, top_features = models[target]
plot_actual_vs_predicted_timeseries(df, model, random_enbid_pci, target, top_features)

print(f"Visualization complete for enbid_pci {random_enbid_pci}")