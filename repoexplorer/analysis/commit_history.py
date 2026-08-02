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
import polars as pl


# Default date range: Jan 20, 2021 to Feb 25, 2026
DEFAULT_START = datetime(2021, 3, 1, tzinfo=timezone.utc)
DEFAULT_END = datetime(2026, 1, 31, tzinfo=timezone.utc)


def _to_utc_datetime(df: pl.DataFrame, col: str) -> pl.Series:
    """
    Coerce a column to UTC Datetime, handling string and existing datetime dtypes.
    Returns a Series of dtype Datetime[us, UTC] with nulls where parsing fails.
    """
    dtype = df.schema[col]

    if dtype == pl.Utf8 or dtype == pl.String:
        series = df[col].str.to_datetime(format=None, strict=False, time_unit="us")
    elif isinstance(dtype, pl.Datetime):
        series = df[col].cast(pl.Datetime("us"), strict=False)
    else:
        # Fallback: cast through string
        series = df[col].cast(pl.Utf8, strict=False).str.to_datetime(format=None, strict=False, time_unit="us")

    # Ensure UTC timezone
    if series.dtype.time_zone is None:
        series = series.dt.replace_time_zone("UTC")
    else:
        series = series.dt.convert_time_zone("UTC")

    return series


def plot_commit_history(
    filtered_data,
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
    filtered_data : polars.DataFrame
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

    if not isinstance(filtered_data, pl.DataFrame):
        filtered_data = pl.DataFrame(filtered_data)

    if filtered_data.is_empty() or "date" not in filtered_data.columns:
        ax.set_title("No commit data", fontsize=title_size)
        return ax

    total_repositories = filtered_data.height

    # Parse date column to UTC Datetime
    date_series = _to_utc_datetime(filtered_data, "date")
    df = pl.DataFrame({"date": date_series}).drop_nulls("date")

    if df.is_empty():
        ax.set_title("No valid dates in commits", fontsize=title_size)
        return ax

    start_dt = start_date if start_date is not None else DEFAULT_START
    end_dt = end_date if end_date is not None else DEFAULT_END
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    df = df.filter(
        (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
    )

    if df.is_empty():
        ax.set_title("No commits in the selected period", fontsize=title_size)
        return ax

    # Aggregate total commits per calendar month
    monthly = (
        df
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ])
        .group_by(["year", "month"])
        .agg(pl.len().alias("count"))
        .sort(["year", "month"])
    )

    # Span every month between first and last with data (fill zeros for gaps)
    rows = monthly.iter_rows(named=True)
    first = next(rows, None)
    if first is None:
        ax.set_title("No commits in the selected period", fontsize=title_size)
        return ax

    # Rebuild as a lookup dict for quick access
    counts_lookup = {(r["year"], r["month"]): r["count"] for r in monthly.iter_rows(named=True)}
    first_year, first_month = monthly["year"][0], monthly["month"][0]
    last_year, last_month = monthly["year"][-1], monthly["month"][-1]

    # Walk every (year, month) in the range
    all_months: list[tuple[int, int]] = []
    y, m = first_year, first_month
    while (y, m) <= (last_year, last_month):
        all_months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    counts_list = [counts_lookup.get((y, m), 0) for y, m in all_months]
    month_dates = [datetime(y, m, 1, tzinfo=timezone.utc) for y, m in all_months]

    ax.plot(
        month_dates,
        counts_list,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="#0d6efd",
    )
    ax.set_xlim(month_dates[0], month_dates[-1])
    # Show all monthly points, but label only every 3 months
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
    df = pl.read_parquet("Data/parquet/commits_combined.parquet")
    plot_commit_history(df)
    plt.show()


if __name__ == "__main__":
    main()
