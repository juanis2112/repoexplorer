#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot commit activity over time from a commits DataFrame.

Expects a DataFrame with a "date" column. Aggregates total commits per month
between a start and end date (default Jan 20, 2021 to Feb 25, 2026) and shows
x-axis labels every three months to reduce clutter.
"""

from datetime import datetime, timezone

import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import pandas as pd


# Default date range: Jan 20, 2021 to Feb 25, 2026
DEFAULT_START = datetime(2021, 3, 1, tzinfo=timezone.utc)
DEFAULT_END = datetime(2026, 1, 31, tzinfo=timezone.utc)


def plot_commit_history(
    filtered_data: pd.DataFrame,
    ax=None,
    start_date=None,
    end_date=None,
    title_prefix="",
    label_size=25,
    title_size=24,
):
    """
    Line plot of total commit counts per calendar month in a date range.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        Commits data with at least a "date" column.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    start_date : datetime, optional
        Only count commits on or after this date. Default Jan 20, 2021.
    end_date : datetime, optional
        Only count commits on or before this date. Default Feb 25, 2026.
    title_prefix : str, optional
        Prefix for the plot title.
    label_size, title_size : int
        Font sizes for axis labels and title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if filtered_data.empty or "date" not in filtered_data.columns:
        ax.set_title("No commit data", fontsize=title_size)
        return ax

    # Count how many repositories are represented in the input.
    # Prefer common repository identifier columns when available.
    repo_id_cols = ["full_name", "name", "repo_name", "repository"]
    repo_col = next((c for c in repo_id_cols if c in filtered_data.columns), None)
    total_repositories = len(filtered_data)
    df = filtered_data[["date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        ax.set_title("No valid dates in commits", fontsize=title_size)
        return ax

    start_ts = pd.Timestamp(start_date if start_date is not None else DEFAULT_START)
    end_ts = pd.Timestamp(end_date if end_date is not None else DEFAULT_END)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if df.empty:
        ax.set_title("No commits in the selected period", fontsize=title_size)
        return ax

    # Aggregate total commits per calendar month using PeriodIndex
    month_periods = df["date"].dt.to_period("M")
    counts = month_periods.value_counts().sort_index()

    # Ensure we have an entry (possibly zero) for every month
    # between the first and last month that actually have commits
    first_period = counts.index.min()
    last_period = counts.index.max()
    all_periods = pd.period_range(start=first_period, end=last_period, freq="M")
    counts = counts.reindex(all_periods, fill_value=0)

    # Convert periods to timestamps for plotting
    month_index = counts.index.to_timestamp()
    ax.plot(
        month_index,
        counts.values,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="#0d6efd",
    )
    ax.set_xlim(month_index.min(), month_index.max())
    # Show all monthly points, but label only every 3 months
    # Anchor labels to March/June/September/December so first is 2021-03
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.set_xlabel("Time", fontsize=label_size)
    ax.set_ylabel("Number of commits", fontsize=label_size)


    title = f"Commit activity by month (Total repositories: {total_repositories})"
    ax.set_title(title, fontsize=title_size)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def main():
    """Load commits from Data/parquet/commits_combined.parquet and plot commit history."""
    df = pd.read_parquet("Data/parquet/commits_combined.parquet")
    plot_commit_history(df)
    plt.show()


if __name__ == "__main__":
    main()
