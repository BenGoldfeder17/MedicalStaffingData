#!/usr/bin/env python3
# correlation_scraper_config.py

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import zscore
import dateutil.parser
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (location- and data-dependent defaults)
STATE              = 'IL'            # two-letter state code for ILI data
WEEKS              = 52              # how many weeks of ILI history to fetch
LAT, LON           = 41.8781, -87.6298  # Chicago, IL coordinates
LOCATION_SLUG      = 'il--chicago'   # Eventbrite city slug for Chicago
HOSPITAL_CSV_PATH  = 'hospital_data.csv'  # default path to hospital CSV
# ─────────────────────────────────────────────────────────────────────────────

def fetch_influenza_data(state=STATE, weeks=WEEKS):
    """
    Fetch weekly influenza-like-illness (ILI) percentages via CDC FluView.
    """
    url = "https://gis.cdc.gov/grasp/fluview/FluViewChartService.svc/ChartData"
    today = datetime.today()
    start_date = (today - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
    params = {
        "Region": state,
        "measure": "ILI_Pct",
        "startWeek": start_date,
        "endWeek": today.strftime("%Y-%m-%d"),
    }
    resp = requests.get(url, params=params)
    data = resp.json().get("ChartData", [])
    df = pd.DataFrame(data)
    df["weekEndingDate"] = pd.to_datetime(df["WeekEndingDate"])
    df = df[["weekEndingDate", "Value"]].rename(columns={"Value": "ili_pct"})
    return df.set_index("weekEndingDate")


def fetch_weather_data(lat=LAT, lon=LON):
    """
    Use NOAA/NWS public API to get recent observations by gridpoint.
    """
    # 1) Discover grid
    pts = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()
    gridId, gridX, gridY = (
        pts["properties"]["gridId"],
        pts["properties"]["gridX"],
        pts["properties"]["gridY"],
    )

    # 2) Retrieve observations
    obs_url = f"https://api.weather.gov/gridpoints/{gridId}/{gridX},{gridY}/observations"
    obs = requests.get(obs_url).json().get("features", [])

    records = []
    for f in obs:
        p = f["properties"]
        t = p.get("temperature", {}).get("value")
        pr = p.get("precipitationLastHour", {}).get("value", 0)
        ts = p.get("timestamp")
        if t is None or ts is None:
            continue
        d = pd.to_datetime(ts).date()
        records.append({"date": d, "temp": t, "precip": pr})
    df = pd.DataFrame(records).groupby("date").agg({"temp": "mean", "precip": "sum"})
    return df


def fetch_community_events(location_slug=LOCATION_SLUG, days_out=30):
    """
    Scrape upcoming Eventbrite events for the given city slug.
    """
    url = f"https://www.eventbrite.com/d/{location_slug}/events/"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    cutoff = datetime.now().date() + timedelta(days=days_out)
    for card in soup.select("div.eds-event-card-content__primary-content"):
        title_el = card.select_one("div.eds-event-card-content__content__principal")
        date_el  = card.select_one("div.eds-text-bs--fixed")
        if not title_el or not date_el:
            continue
        title = title_el.get_text(strip=True)
        try:
            dt = dateutil.parser.parse(date_el.get_text(strip=True)).date()
        except Exception:
            continue
        if dt <= cutoff:
            events.append({"date": dt, "event": title})
    if not events:
        return pd.DataFrame(columns=["count_events"])
    df = pd.DataFrame(events)
    df["count_events"] = 1
    return df.groupby("date")[["count_events"]].sum()


def fetch_hospital_data(csv_path):
    """
    Load daily staffing and patient volumes from local CSV.
    Expects columns: date, staffing_need, patient_count
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df.set_index("date")


def merge_datasets(ili, weather, events, hospital):
    """
    Merge all sources on date and fill gaps.
    """
    df = pd.concat([ili, weather, events, hospital], axis=1)
    return df.ffill().bfill()


def detect_anomalies(df, column, z_thresh=2.5):
    """
    Flag rows where the z-score of `column` exceeds the threshold.
    """
    df[f"{column}_z"] = zscore(df[column].fillna(0))
    return df[abs(df[f"{column}_z"]) > z_thresh]


def compute_correlations(df, feature_cols, target_cols):
    """
    Return Pearson correlations between features and targets.
    """
    return df[feature_cols + target_cols].corr().loc[feature_cols, target_cols]


def main():
    parser = argparse.ArgumentParser(
        description="Scrape public data and correlate anomalies (no API keys)."
    )
    parser.add_argument(
        "--hospital_csv", default=HOSPITAL_CSV_PATH,
        help="Path to CSV with columns: date, staffing_need, patient_count"
    )
    parser.add_argument(
        "--state", default=STATE,
        help="State code for ILI data (default: IL)"
    )
    parser.add_argument(
        "--weeks", type=int, default=WEEKS,
        help="Weeks of ILI history (default: 52)"
    )
    parser.add_argument(
        "--lat", type=float, default=LAT,
        help="Latitude for weather (default: Chicago)"
    )
    parser.add_argument(
        "--lon", type=float, default=LON,
        help="Longitude for weather (default: Chicago)"
    )
    parser.add_argument(
        "--location_slug", default=LOCATION_SLUG,
        help="Eventbrite city slug (default: il--chicago)"
    )
    args = parser.parse_args()

    # 1. Fetch each dataset
    ili_df     = fetch_influenza_data(state=args.state, weeks=args.weeks)
    weather_df = fetch_weather_data(lat=args.lat, lon=args.lon)
    events_df  = fetch_community_events(location_slug=args.location_slug)
    hosp_df    = fetch_hospital_data(args.hospital_csv)

    # 2. Merge all sources
    master = merge_datasets(ili_df, weather_df, events_df, hosp_df)

    # 3. Detect staffing & patient anomalies
    staff_anom   = detect_anomalies(master, "staffing_need")
    patient_anom = detect_anomalies(master, "patient_count")

    # 4. Compute correlations
    features = ["ili_pct", "temp", "precip", "count_events"]
    targets  = ["staffing_need", "patient_count"]
    corr = compute_correlations(master, features, targets)

    # 5. Output results
    print("\n=== Correlation Matrix ===")
    print(corr.to_string(float_format="{:.2f}".format))

    print("\n=== Staffing Anomalies ===")
    print(staff_anom[["staffing_need", "staffing_need_z"]].to_string())

    print("\n=== Patient Count Anomalies ===")
    print(patient_anom[["patient_count", "patient_count_z"]].to_string())


if __name__ == "__main__":
    main()
