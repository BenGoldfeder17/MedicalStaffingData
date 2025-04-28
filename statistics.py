#!/usr/bin/env python3
# compare_predictions.py

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def load_data(pred_path, actual_path, date_col='date'):
    """
    Load prediction and actual CSVs, parse dates, and merge on date_col.
    """
    df_pred = pd.read_csv(pred_path, parse_dates=[date_col])
    df_act  = pd.read_csv(actual_path, parse_dates=[date_col])
    df = pd.merge(df_pred, df_act, on=date_col, how='inner', suffixes=('_pred', '_act'))
    df.set_index(date_col, inplace=True)
    return df

def compute_metrics(y_true, y_pred):
    """
    Compute common regression/forecasting error metrics.
    Returns a dict with MAE, RMSE, MAPE, MBE, and R².
    """
    # avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    
    metrics = {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE (%)':  mape,
        'MBE':  np.mean(y_pred - y_true),             # Mean Bias Error
        'R2':   r2_score(y_true, y_pred)
    }
    return metrics

def compare(df, output_csv=None):
    """
    For each pair of columns (col_pred, col_act) where col_pred endswith '_pred'
    and corresponding col_act exists, compute metrics.
    """
    results = []
    for col in df.columns:
        if col.endswith('_pred'):
            base = col[:-5]  # strip '_pred'
            act_col = base + '_act'
            if act_col in df.columns:
                y_pred = df[col]
                y_true = df[act_col]
                m = compute_metrics(y_true, y_pred)
                results.append({
                    'Metric': base,
                    **m
                })
    result_df = pd.DataFrame(results).set_index('Metric')
    print(result_df.to_string(float_format='{:,.2f}'.format))
    if output_csv:
        result_df.to_csv(output_csv)
        print(f"\nSaved metrics to {output_csv}")
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="Compare predicted vs. actual data and print error metrics."
    )
    parser.add_argument(
        '--pred', required=True,
        help="Path to CSV with predictions; must have columns 'date' and '<feature>_pred'."
    )
    parser.add_argument(
        '--actual', required=True,
        help="Path to CSV with actuals; must have columns 'date' and '<feature>_act'."
    )
    parser.add_argument(
        '--date_col', default='date',
        help="Name of the date column in both CSVs (default: 'date')."
    )
    parser.add_argument(
        '--out', default=None,
        help="If provided, path to write metrics CSV."
    )
    args = parser.parse_args()

    df = load_data(args.pred, args.actual, date_col=args.date_col)
    compare(df, output_csv=args.out)

if __name__ == '__main__':
    main()
