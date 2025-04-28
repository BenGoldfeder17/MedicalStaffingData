#!/usr/bin/env python3
# hospital_staffing_tf.py

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import dateutil.parser
from scipy.stats import zscore
import argparse

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration defaults
STATE = 'IL'
WEEKS = 52
LAT, LON = 41.8781, -87.6298
LOCATION_SLUG = 'il--chicago'

# --- Specialist ranking ---
def load_specialist_data(path):
    return pd.read_csv(path, parse_dates=['certified_date'])

def engineer_specialist_features(df):
    df['vacation_rate']     = df['vacation_days'] / df['total_work_days']
    df['absenteeism_rate']  = df['absent_days']  / df['total_work_days']
    df['success_rate']      = df['successful_cases'] / df['total_cases']
    df['experience_yrs']    = (pd.to_datetime('today') - df['certified_date']).dt.days / 365.0
    df['avg_case_duration'] = df['case_duration_minutes_mean']
    df['complication_cost'] = df['complication_rate'] * df['cost_per_complication']
    df['readmission_cost']  = df['readmission_rate']  * df['cost_per_readmission']
    df['staffing_cost']     = (
        df['RN_hours'] * df['RN_rate'] +
        df['tech_hours'] * df['tech_rate'] +
        df.get('assistant_hours', 0) * df.get('assistant_rate', 0)
    )
    df['base_revenue']      = df['total_cases'] * df['avg_reimbursement']
    df['penalty_cost']      = df['readmission_rate'] * df['readmission_penalty']
    df['net_margin']        = (
        df['base_revenue']
        - df['complication_cost']
        - df['readmission_cost']
        - df['staffing_cost']
        - df['penalty_cost']
    )
    df['cost_effectiveness'] = df['net_margin'] / df['staffing_cost'].replace(0, np.nan)
    return df

# --- Time-series data ---
def fetch_influenza_data(state=STATE, weeks=WEEKS):
    url = "https://gis.cdc.gov/grasp/fluview/FluViewChartService.svc/ChartData"
    today = datetime.today()
    start = (today - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
    params = {"Region": state, "measure": "ILI_Pct",
              "startWeek": start, "endWeek": today.strftime("%Y-%m-%d")}
    data = requests.get(url, params=params).json().get("ChartData", [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['WeekEndingDate']).dt.date
    return df[['date', 'Value']].rename(columns={'Value':'ili_pct'}).set_index('date')

def fetch_weather_data(lat=LAT, lon=LON):
    pts = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()['properties']
    grid = f"{pts['gridId']}/{pts['gridX']},{pts['gridY']}"
    obs = requests.get(f"https://api.weather.gov/gridpoints/{grid}/observations").json().get("features", [])
    records = []
    for f in obs:
        p = f['properties']
        t = p.get('temperature', {}).get('value')
        pr = p.get('precipitationLastHour', {}).get('value', 0)
        ts = p.get('timestamp')
        if t is None or ts is None:
            continue
        d = pd.to_datetime(ts).date()
        records.append({'date': d, 'temp': t, 'precip': pr})
    df = pd.DataFrame(records).groupby('date').agg({'temp':'mean', 'precip':'sum'})
    return df

def fetch_community_events(slug=LOCATION_SLUG, days_out=30):
    url = f"https://www.eventbrite.com/d/{slug}/events/"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    cutoff = datetime.now().date() + timedelta(days=days_out)
    recs = []
    for card in soup.select("div.eds-event-card-content__primary-content"):
        title_el = card.select_one("div.eds-event-card-content__content__principal")
        date_el  = card.select_one("div.eds-text-bs--fixed")
        if not title_el or not date_el:
            continue
        try:
            d = dateutil.parser.parse(date_el.get_text(strip=True)).date()
        except:
            continue
        if d <= cutoff:
            recs.append({'date': d, 'event': title_el.get_text(strip=True)})
    if not recs:
        return pd.DataFrame(columns=['count_events'])
    df = pd.DataFrame(recs)
    df['count_events'] = 1
    return df.groupby('date')[['count_events']].sum()

# --- Hospital & procedure data ---
def load_hospital_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = df['date'].dt.date
    return df.set_index('date')

def load_procedure_data(path):
    return pd.read_csv(path, parse_dates=['date'])

# --- Merge ---
def merge_time_series(ili, weather, events, hospital):
    df = pd.concat([ili, weather, events, hospital], axis=1).ffill().bfill().reset_index().rename(columns={'index':'date'})
    return df

# --- Anomalies & correlation ---
def detect_anomalies(df, col, z_thresh=2.5):
    df[f"{col}_z"] = zscore(df[col].fillna(0))
    return df[np.abs(df[f"{col}_z"]) > z_thresh]

def compute_correlations(df, features, targets):
    return df[features + targets].corr().loc[features, targets]

# --- Build & train Keras model ---
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Hospital Staffing with TensorFlow")
    parser.add_argument('--specialist_csv', required=True)
    parser.add_argument('--hospital_csv',    required=True)
    parser.add_argument('--procedure_csv',   required=True)
    parser.add_argument('--model_out',       default='staffing_model.keras',
                        help='Output .keras model file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pred_csv', default=None)
    parser.add_argument('--actual_csv', default=None)
    args = parser.parse_args()

    # Rank specialists
    spec = load_specialist_data(args.specialist_csv)
    spec = engineer_specialist_features(spec)
    print("Top 5 Cost-Effective Specialists:")
    print(spec.sort_values('cost_effectiveness', ascending=False)[['specialist_id','cost_effectiveness']].head())

    # Time-series merge
    ili     = fetch_influenza_data()
    weather = fetch_weather_data()
    events  = fetch_community_events()
    hosp    = load_hospital_data(args.hospital_csv)
    ts_df   = merge_time_series(ili, weather, events, hosp)

    # Correlation & anomalies
    features_ts = ['ili_pct','temp','precip','count_events']
    targets_ts  = ['staffing_need','patient_count']
    print("Correlation matrix:")
    print(compute_correlations(ts_df, features_ts, targets_ts))
    print("Staffing anomalies:")
    print(detect_anomalies(ts_df, 'staffing_need')[['date','staffing_need','staffing_need_z']])
    print("Patient anomalies:")
    print(detect_anomalies(ts_df, 'patient_count')[['date','patient_count','patient_count_z']])

    # Procedure-level modeling
    proc_df = load_procedure_data(args.procedure_csv)
    df_all  = pd.merge(proc_df, ts_df, on='date', how='left').dropna()

    features = [
        'or_utilization_rate',
        'cleaning_time_variance',
        'turnaround_time_compliance',
        'cost_per_staff_hour'
    ] + features_ts
    target   = 'staffing_need'

    X = df_all[features].values
    y = df_all[target].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2
    )

    # Save as .keras model
    model.save(args.model_out, save_format='keras')
    print(f"Model saved to {args.model_out}")

    # Optional comparison
    if args.pred_csv and args.actual_csv:
        df_pred = pd.read_csv(args.pred_csv, parse_dates=['date'])
        df_act  = pd.read_csv(args.actual_csv, parse_dates=['date'])
        df_cmp  = pd.merge(df_pred, df_act, on='date', suffixes=('_pred','_act')).dropna()
        for col in df_cmp.columns:
            if col.endswith('_pred'):
                base = col[:-5]
                y_p = df_cmp[col].values
                y_a = df_cmp[f"{base}_act"].values
                mae  = mean_absolute_error(y_a, y_p)
                rmse = np.sqrt(mean_squared_error(y_a, y_p))
                r2   = r2_score(y_a, y_p)
                print(f"{base}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")

if __name__ == '__main__':
    main()
