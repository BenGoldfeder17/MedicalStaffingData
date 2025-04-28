#!/usr/bin/env python3
# cost_effective_specialists.py

import pandas as pd
import argparse

def load_data(metrics_path):
    """
    Load specialist metrics from a CSV file.
    """
    return pd.read_csv(metrics_path)

def engineer_features(df):
    """
    Compute derived features and cost/revenue/margin metrics.
    """
    # Availability & performance
    df['vacation_rate']     = df['vacation_days'] / df['total_work_days']
    df['absenteeism_rate']  = df['absent_days']  / df['total_work_days']
    df['success_rate']      = df['successful_cases'] / df['total_cases']
    df['experience_yrs']    = (
        pd.to_datetime('today') - pd.to_datetime(df['certified_date'])
    ).dt.days / 365.0
    df['avg_case_duration'] = df['case_duration_minutes_mean']

    # Cost components
    df['complication_cost'] = df['complication_rate'] * df['cost_per_complication']
    df['readmission_cost']  = df['readmission_rate']  * df['cost_per_readmission']
    df['staffing_cost']     = (
        df['RN_hours']   * df['RN_rate']   +
        df['tech_hours'] * df['tech_rate']  +
        df.get('assistant_hours', 0) * df.get('assistant_rate', 0)
    )

    # Revenue components
    df['base_revenue'] = df['total_cases'] * df['avg_reimbursement']
    df['penalty_cost'] = df['readmission_rate'] * df['readmission_penalty']

    # Net margin per period
    df['net_margin'] = (
        df['base_revenue']
        - df['complication_cost']
        - df['readmission_cost']
        - df['staffing_cost']
        - df['penalty_cost']
    )

    # Cost-effectiveness: margin per dollar of staffing cost
    df['cost_effectiveness'] = df['net_margin'] / df['staffing_cost'].replace(0, pd.NA)

    return df

def rank_specialists(df, top_n=None):
    """
    Sort specialists by cost-effectiveness and return top N (or all if None).
    """
    ranked = df.sort_values(
        by='cost_effectiveness',
        ascending=False
    )
    if top_n:
        ranked = ranked.head(top_n)
    return ranked

def main():
    parser = argparse.ArgumentParser(
        description='Identify most cost-effective specialists.'
    )
    parser.add_argument(
        '--metrics', required=True,
        help='Path to CSV file with specialist metrics.'
    )
    parser.add_argument(
        '--top', type=int, default=None,
        help='Number of top specialists to show (default: all).'
    )
    parser.add_argument(
        '--out', default=None,
        help='Optional path to write ranked results as CSV.'
    )
    args = parser.parse_args()

    # Load & prepare data
    df = load_data(args.metrics)
    df = engineer_features(df)

    # Columns to display
    display_cols = [
        'specialist_id',
        'cost_effectiveness',
        'net_margin',
        'base_revenue',
        'staffing_cost',
        'complication_rate',
        'readmission_rate',
        'success_rate',
        'avg_case_duration',
        'vacation_rate',
        'absenteeism_rate'
    ]

    # Rank and select
    ranked = rank_specialists(df, top_n=args.top)

    # Print to console
    print(ranked[display_cols].to_string(index=False, float_format='{:,.2f}'.format))

    # Optionally write to CSV
    if args.out:
        ranked.to_csv(args.out, index=False)
        print(f"\nRanked results saved to: {args.out}")

if __name__ == '__main__':
    main()

