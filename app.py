# app.py
# Uber NCR 2024 — Analytics & Decision Lab (Lean, Py3.13-friendly, Optimized)
# No scikit-learn / shap / prophet / xgboost / lightgbm.
# GLM (classification), ARIMA (forecast), NumPy KMeans (clustering), OLS (regression).
from __future__ import annotations

import os
import io
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------#
# App Config
# ------------------------------#
st.set_page_config(
    page_title="Uber NCR 2024 — Analytics & Decision Lab (Lean)",
    page_icon="🚖",
    layout="wide",
)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Schema (strict headers)
SCHEMA = [
    "Date", "Time", "Booking ID", "Booking Status", "Customer ID", "Vehicle Type",
    "Pickup Location", "Drop Location", "Avg VTAT", "Avg CTAT",
    "Cancelled Rides by Customer", "Reason for cancelling by Customer",
    "Cancelled Rides by Driver", "Driver Cancellation Reason",
    "Incomplete Rides", "Incomplete Rides Reason",
    "Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating", "Payment Method"
]
CANONICAL_STATUSES = ["Completed", "Customer Cancelled", "Driver Cancelled", "No Driver Found", "Incomplete"]

COLORS = {
    "insight": "#5e60ce",
    "demand": "#1f77b4",
    "risk": "#e76f51",
    "finance": "#2a9d8f",
    "cx": "#9b5de5",
}

# ------------------------------#
# Helpers
# ------------------------------#
def _title_case_or_nan(x) -> Optional[str]:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s == "0" or s.lower() in {"na", "nan", "none", "null"}:
        return np.nan
    return s.title()

def time_bucket_from_hour(h: int) -> str:
    if 5 <= h < 12:
        return "Morning (05–11)"
    if 12 <= h < 17:
        return "Afternoon (12–16)"
    if 17 <= h < 21:
        return "Evening (17–20)"
    return "Night (21–04)"

def compress_categories(s: pd.Series, top_n: int = 30, other_label: str = "Other") -> pd.Series:
    s = s.astype("string").fillna("Unknown")
    counts = s.value_counts(dropna=False)
    top = counts.nlargest(top_n).index
    return s.where(s.isin(top), other_label)

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# ------------------------------#
# Data I/O & Processing
# ------------------------------#
@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO | str) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

@st.cache_data(show_spinner="Processing data...")
def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    msgs: List[str] = []
    df = df.copy()

    # MORE ROBUST: Let Pandas infer the date format automatically.
    # The 'dayfirst=True' argument helps it guess correctly (e.g., DD/MM/YYYY).
    df['timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        dayfirst=True,
        errors='coerce'
    )
    
    bad = int(df["timestamp"].isna().sum())
    if bad:
        msgs.append(f"⚠️ Dropped {bad} rows with invalid Date/Time.")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    
    # If all rows were dropped, return the empty dataframe early.
    if df.empty:
        return df, msgs

    # Features
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["time_bucket"] = df["hour"].apply(time_bucket_from_hour)

    # Numeric casts
    rename_map = {
        "Avg VTAT": "avg_vtat", "Avg CTAT": "avg_ctat",
        "Cancelled Rides by Customer": "cancelled_by_customer",
        "Cancelled Rides by Driver": "cancelled_by_driver",
        "Incomplete Rides": "incomplete_rides", "Booking Value": "booking_value",
        "Ride Distance": "ride_distance", "Driver Ratings": "driver_ratings",
        "Customer Rating": "customer_rating",
    }
    for src, dst in rename_map.items():
        df[dst] = safe_numeric(df[src])

    # Clean reason text
    df["reason_customer"]  = df["Reason for cancelling by Customer"].map(_title_case_or_nan)
    df["reason_driver"]    = df["Driver Cancellation Reason"].map(_title_case_or_nan)
    df["reason_incomplete"]= df["Incomplete Rides Reason"].map(_title_case_or_nan)

    # OPTIMIZED: Vectorized booking status creation
    raw_status_lower = df["Booking Status"].astype(str).str.lower().str.strip()
    
    conditions = [
        raw_status_lower == "completed",
        raw_status_lower.str.contains("no driver found", na=False),
        raw_status_lower.str.contains("incomplete", na=False) | df["incomplete_rides"].gt(0),
        raw_status_lower.str.contains("customer", na=False) | df["cancelled_by_customer"].gt(0),
        raw_status_lower.str.contains("driver", na=False) | df["cancelled_by_driver"].gt(0)
    ]
    choices = ["Completed", "No Driver Found", "Incomplete", "Customer Cancelled", "Driver Cancelled"]
    df["booking_status_canon"] = np.select(conditions, choices, default=df["Booking Status"])
    df["will_complete"] = (df["booking_status_canon"] == "Completed").astype(int)

    # Make categorical
    for c in ["Booking Status", "booking_status_canon", "Vehicle Type", "Pickup Location",
              "Drop Location", "Payment Method", "time_bucket"]:
        df[c] = df[c].astype("category")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, msgs

def find_default_csv() -> Optional[str]:
    if os.path.exists("ncr_ride_bookingsv1.csv"):
        return "ncr_ride_bookingsv1.csv"
    return None

# --- The rest of the functions (UI helpers, ML helpers) are unchanged ---

# ... (rest of your code, like bar_from_series, filter_block, etc.)
def bar_from_series(series: pd.Series, title: str, x_label: str = None, y_label: str = "Count", color=None):
    color = color or COLORS["demand"]
    dfp = series.reset_index()
    dfp.columns = [x_label or series.index.name or "Category", y_label]
    fig = px.bar(dfp, x=dfp.columns[0], y=dfp.columns[1], title=title, color_discrete_sequence=[color])
    st.plotly_chart(fig, use_container_width=True)

def filter_block(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("📅 Global Filters")
    min_d, max_d = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    drange = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(drange, tuple) and len(drange) == 2:
        start_date, end_date = drange
        df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]

    vtypes = st.sidebar.multiselect("Vehicle Type", sorted(df["Vehicle Type"].dropna().unique().tolist()))
    if vtypes: df = df[df["Vehicle Type"].isin(vtypes)]

    pm = st.sidebar.multiselect("Payment Method", sorted(df["Payment Method"].dropna().unique().tolist()))
    if pm: df = df[df["Payment Method"].isin(pm)]

    bs = st.sidebar.multiselect("Booking Status (canonical)", sorted(df["booking_status_canon"].dropna().unique().tolist()))
    if bs: df = df[df["booking_status_canon"].isin(bs)]

    pls = st.sidebar.multiselect("Pickup Location", sorted(df["Pickup Location"].dropna().unique().tolist()))
    if pls: df = df[df["Pickup Location"].isin(pls)]

    dls = st.sidebar.multiselect("Drop Location", sorted(df["Drop Location"].dropna().unique().tolist()))
    if dls: df = df[df["Drop Location"].isin(dls)]

    return df

def empty_state(df: pd.DataFrame) -> bool:
    if df.empty:
        st.info("No rows match the current filters. Adjust filters in the sidebar.")
        return True
    return False

# ------------------------------#
# Main App Logic
# ------------------------------#
def main():
    st.sidebar.title("🚖 Uber NCR 2024 — Analytics & Decision Lab (Lean)")
    st.sidebar.markdown("**Data Source**")
    source = st.sidebar.radio("Choose source", ["Auto-detect file", "Upload CSV"], index=0)

    df_raw = None
    try:
        if source == "Auto-detect file":
            path = find_default_csv()
            if path:
                df_raw = load_csv(path)
            else:
                st.sidebar.warning("Default file not found. Please upload the CSV.")
        if df_raw is None:
            uploaded = st.sidebar.file_uploader("Upload `ncr_ride_bookingsv1.csv`", type=["csv", "txt"])
            if uploaded:
                df_raw = load_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to load CSV: {e}")
        st.stop()

    if df_raw is None:
        st.info("Please upload or select a data source to begin.")
        st.stop()

    # --- NEW: DEBUGGING METRICS ---
    # We will add two columns to show the row counts before and after processing.
    col1, col2 = st.columns(2)
    col1.metric("Rows Loaded from CSV", f"{len(df_raw):,}")
    
    df, load_msgs = preprocess(df_raw)
    
    col2.metric("Rows After Preprocessing", f"{len(df):,}")
    # --- END OF DEBUGGING METRICS ---
    
    for m in load_msgs:
        st.warning(m)

    # Handle case where preprocessing results in an empty DataFrame
    if df.empty:
        st.error("All data was removed during preprocessing. Please check the data file for issues with 'Date' and 'Time' columns.")
        st.stop()

    df_f = filter_block(df)
    if empty_state(df_f):
        st.stop()

    # The rest of your app's main function...
    st.sidebar.download_button("Download Filtered Data (CSV)", df_f.to_csv(index=False).encode("utf-8"),
                               "filtered_data.csv", "text/csv")
    # ... etc.

# (Your original `main` function continues here, with all the tabs)
# (For brevity, I'm omitting the tab creation code as it doesn't need to change)
# (Make sure to paste this into your existing full script)
    tab_names = [
        "1) Executive Overview", "2) Ride Completion & Cancellation", "3) Geographical & Temporal",
        "4) Operational Efficiency", "5) Financial Analysis", "6) Ratings & Satisfaction",
        "7) Incomplete Rides", "8) ML Lab", "9) Risk & Fraud",
        "10) Operations Simulator", "11) Reports & Exports",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("## Executive Overview")
        total = len(df_f)
        if total > 0:
            comp = int((df_f["booking_status_canon"] == "Completed").sum())
            cust_cxl = int((df_f["booking_status_canon"] == "Customer Cancelled").sum())
            drv_cxl = int((df_f["booking_status_canon"] == "Driver Cancelled").sum())
            avg_drv = df_f["driver_ratings"].replace(0, np.nan).mean()
            avg_cus = df_f["customer_rating"].replace(0, np.nan).mean()
            revenue = df_f.loc[df_f["will_complete"] == 1, "booking_value"].sum()

            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Total Bookings", f"{total:,}")
            c2.metric("Completion %", f"{(comp/total*100):.1f}%")
            # ... rest of your tab code
    
    # ... all other tabs

if __name__ == "__main__":
    main()
