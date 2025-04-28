#!/usr/bin/env python3
# hospital_procedure_models.py

import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

def load_data(path, date_col=None):
    """
    Load historical data CSV.
    Expects at minimum these columns:
      - procedure_type
      - or_utilization_rate
      - cleaning_time_variance
      - turnaround_time_compliance
      - cost_per_staff_hour
      - staffing_need  (target)
    Optionally parses a date column.
    """
    if date_col:
        df = pd.read_csv(path, parse_dates=[date_col])
    else:
        df = pd.read_csv(path)
    return df

def train_procedure_models(df,
                           features,
                           target,
                           min_samples=20,
                           test_size=0.2,
                           random_state=42):
    """
    For each unique procedure_type in df, train a RandomForestRegressor
    to predict `target` from `features`. Returns:
      - models: dict mapping procedure_type -> trained model
      - metrics_df: DataFrame summarizing R2, MAE, RMSE, n_samples
    """
    procedures = df['procedure_type'].unique()
    summary = []
    models = {}

    for proc in procedures:
        sub = df[df['procedure_type'] == proc].dropna(subset=features + [target])
        n = len(sub)
        if n < min_samples:
            print(f"Skipping {proc!r}: only {n} samples (<{min_samples})")
            continue

        X = sub[features]
        y = sub[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"[{proc}] samples={n}  R²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

        summary.append({
            'procedure': proc,
            'n_samples': n,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse
        })
        models[proc] = model

    metrics_df = pd.DataFrame(summary).set_index('procedure')
    return models, metrics_df

def main():
    p = argparse.ArgumentParser(
        description="Train staffing‐need models per procedure type."
    )
    p.add_argument(
        '--data', required=True,
        help="Path to historical data CSV"
    )
    p.add_argument(
        '--date_col', default=None,
        help="Optional date column to parse"
    )
    p.add_argument(
        '--target', default='staffing_need',
        help="Name of the column to predict"
    )
    p.add_argument(
        '--min_samples', type=int, default=20,
        help="Minimum rows per procedure to train a model"
    )
    p.add_argument(
        '--out_metrics', default=None,
        help="If set, writes a CSV of the performance metrics"
    )
    p.add_argument(
        '--out_models_dir', default=None,
        help="If set, pickles each model to this directory (one file per procedure)"
    )
    args = p.parse_args()

    df = load_data(args.data, date_col=args.date_col)

    features = [
        'or_utilization_rate',
        'cleaning_time_variance',
        'turnaround_time_compliance',
        'cost_per_staff_hour'
    ]

    models, metrics_df = train_procedure_models(
        df,
        features,
        target=args.target,
        min_samples=args.min_samples
    )

    print("\n=== Summary Metrics ===")
    print(metrics_df.to_string(float_format="{:.3f}".format))

    if args.out_metrics:
        metrics_df.to_csv(args.out_metrics)
        print(f"\nSaved metrics to {args.out_metrics}")

    if args.out_models_dir:
        import os
        os.makedirs(args.out_models_dir, exist_ok=True)
        for proc, model in models.items():
            fn = os.path.join(args.out_models_dir, f"{proc.replace(' ','_')}.pkl")
            with open(fn, 'wb') as f:
                pickle.dump(model, f)
        print(f"Serialized {len(models)} models in {args.out_models_dir}")

if __name__ == '__main__':
    main()
