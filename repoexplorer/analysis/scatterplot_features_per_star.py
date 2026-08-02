#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def _make_bins(max_stars: int, bins: int):
    """Return (breaks, labels, bin_centers) for `bins` equal-width buckets over [0, max_stars)."""
    edges = np.linspace(0, max_stars, bins + 1)
    labels = [f"[{edges[i]:.0f}, {edges[i+1]:.0f})" for i in range(bins)]
    centers = (edges[:-1] + edges[1:]) / 2
    return edges[1:-1].tolist(), labels, centers


def _bin_counts(df: pl.DataFrame, star_col: str, labels: list[str], breaks: list[float]) -> dict[str, int]:
    """Group df by star bin and return label -> count dict."""
    binned = df.with_columns(
        pl.col(star_col)
        .cut(breaks=breaks, labels=labels, left_closed=True)
        .cast(pl.Utf8)
        .alias("_bin")
    )
    counts = (
        binned
        .group_by("_bin")
        .agg(pl.len().alias("n"))
    )
    return {row["_bin"]: row["n"] for row in counts.iter_rows(named=True)}


def plot_feature_presence_by_stars_grid(
    df, features, star_col='stargazers_count', max_stars=1000,
    bins=5, figsize=(18, 5), tick_size=16,
    label_size=20, title_size=24, annotations_size=16
    ):

    """
    Plot the percentage of repositories with specific features across star count bins.

    This function creates a grid of scatter plots, one for each feature, showing the
    percentage of repositories containing that feature within predefined star count bins.
    A linear regression line is included to visualize trends.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing repository metadata, including star counts and feature presence.
    features : list of str
        List of column names corresponding to features (e.g., community files) to evaluate.
    star_col : str, default='stargazers_count'
        Column in `df` representing the number of stars.
    max_stars : int, default=1000
        Maximum number of stars to consider; repositories with more stars are filtered out.
    bins : int, default=5
        Number of bins to divide the star count range into.
    figsize : tuple of int, default=(18, 5)
        Size of the entire figure.
    tick_size : int, default=16
        Font size for tick labels.
    label_size : int, default=20
        Font size for axis labels and subplot titles.
    title_size : int, default=24
        Font size for the overall figure title.
    annotations_size : int, default=16
        (Currently unused) Size for annotations on the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    df = df.filter(pl.col(star_col) <= max_stars)
    total_repositories = df.height

    breaks, labels, bin_centers = _make_bins(max_stars, bins)
    total_dict = _bin_counts(df, star_col, labels, breaks)
    total_array = np.array([total_dict.get(lbl, 0) for lbl in labels], dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        # Repos with the feature present
        df_feature = df.filter(pl.col(feature).is_not_null())
        feat_dict = _bin_counts(df_feature, star_col, labels, breaks)
        feat_array = np.array([feat_dict.get(lbl, 0) for lbl in labels], dtype=float)

        # Compute percentage (handle bins with zero total count)
        percentages = np.where(total_array > 0, feat_array / total_array * 100, 0.0)

        ax.scatter(bin_centers, percentages, alpha=0.7)

        # Linear regression line
        slope, intercept, r_value, p_value, std_err = linregress(bin_centers, percentages)
        line_x = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        line_y = intercept + slope * line_x
        ax.plot(line_x, line_y, color='red', linestyle='--')

        ax.set_title(feature.replace("_", " ").title(), fontsize=label_size)
        ax.set_xlabel("# Stars", fontsize=label_size)
        ax.set_ylabel("Percentage with Feature", fontsize=label_size)
        tick_interval = max_stars // 5
        xticks = np.arange(0, max_stars + 1, tick_interval)
        ax.set_xticks(xticks)
        ax.set_xlim(0, max_stars)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.grid(True)

    suptitle = (
        r"$\bf{Percentage\ of\ Community\ Files\ by\ Number\ of\ Stars\ }$" +
        r"$\bf{DEV\ Repositories}$" + f" (Total: {total_repositories})"
    )
    fig.suptitle(suptitle, fontsize=title_size)

    return fig


def plot_avg_feature_presence_by_stars(
    df, features, star_col='stargazers_count', max_stars=1000,
    bins=20, figsize=(8, 5), tick_size=16,
    label_size=20, title_size=22
    ):

    """
    Plot the average percentage of repositories with given features across star count bins.

    This function computes the average presence of several features across star bins
    and visualizes the trend in a single scatter plot with a linear regression line.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing repository metadata, including star counts and feature presence.
    features : list of str
        List of column names corresponding to features (e.g., community files) to average.
    star_col : str, default='stargazers_count'
        Column in `df` representing the number of stars.
    max_stars : int, default=1000
        Maximum number of stars to consider; repositories with more stars are filtered out.
    bins : int, default=20
        Number of bins to divide the star count range into.
    figsize : tuple of int, default=(8, 5)
        Size of the figure.
    tick_size : int, default=16
        Font size for tick labels.
    label_size : int, default=20
        Font size for axis labels.
    title_size : int, default=22
        Font size for the plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)

    df = df.filter(pl.col(star_col) <= max_stars)
    total_repositories = df.height

    breaks, labels, bin_centers = _make_bins(max_stars, bins)
    total_dict = _bin_counts(df, star_col, labels, breaks)
    total_array = np.array([total_dict.get(lbl, 0) for lbl in labels], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Build matrix of per-feature percentages, then average
    pct_matrix = []
    for feature in features:
        df_feature = df.filter(pl.col(feature).is_not_null())
        feat_dict = _bin_counts(df_feature, star_col, labels, breaks)
        feat_array = np.array([feat_dict.get(lbl, 0) for lbl in labels], dtype=float)
        percentages = np.where(total_array > 0, feat_array / total_array * 100, 0.0)
        pct_matrix.append(percentages)

    avg_percentages = np.mean(pct_matrix, axis=0)

    # Scatter plot
    ax.scatter(bin_centers, avg_percentages, alpha=0.7)

    # Linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(bin_centers, avg_percentages)
    line_x = np.linspace(bin_centers.min(), bin_centers.max(), 100)
    line_y = intercept + slope * line_x
    ax.plot(line_x, line_y, color='red', linestyle='--')

    title = (
        r"$\bf{UC\ Average\ Community\ File\ Presence\ }$" + "\n" +
        r"$\bf{DEV\ Repos}$" + f" (Total: {total_repositories})"
    )
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("# Stars", fontsize=label_size)
    ax.set_ylabel("Average % with Feature", fontsize=label_size)
    tick_interval = max_stars // 5
    xticks = np.arange(0, max_stars + 1, tick_interval)
    ax.set_xticks(xticks)
    ax.set_xlim(0, max_stars)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(True)

    return fig
