#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


_BUCKET_LABELS = ["0-10", "11-100", "101-1000", "1K-10K", "10K+"]
_VALUE_COL = "stargazers_count"


def _values_to_bucket_labels(values: pd.Series) -> pd.Series:
    """Map numeric values to ordered bucket labels; NaN / invalid -> NA."""
    v = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    m = v.notna() & (v >= 0) & (v <= 10)
    out.loc[m] = _BUCKET_LABELS[0]
    m = v.notna() & (v > 10) & (v <= 100)
    out.loc[m] = _BUCKET_LABELS[1]
    m = v.notna() & (v > 100) & (v <= 1000)
    out.loc[m] = _BUCKET_LABELS[2]
    m = v.notna() & (v > 1000) & (v <= 10000)
    out.loc[m] = _BUCKET_LABELS[3]
    m = v.notna() & (v > 10000)
    out.loc[m] = _BUCKET_LABELS[4]
    return out


def plot_stars_distribution_bar(
    filtered_data,
    acronym="",
    ax=None,
    color_map=None,
    title_prefix="",
    hide_ylabel=False,
    ylim=None,
    label_size=25,
    title_size=24,
    textprops=18,
    legend_size=None,
):
    """
    Bar chart: count of repositories per stars bucket (0-10, 11-100, ...).
    """
    if _VALUE_COL not in filtered_data.columns:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        return

    plot_data = filtered_data
    total = len(plot_data)
    buckets = _values_to_bucket_labels(plot_data[_VALUE_COL])
    valid = buckets.notna()
    if not valid.any():
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        return

    counts = buckets[valid].value_counts().reindex(_BUCKET_LABELS, fill_value=0)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(_BUCKET_LABELS))]
    if color_map:
        colors = [color_map.get(lbl, colors[i]) for i, lbl in enumerate(_BUCKET_LABELS)]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors)

    ymax = max(counts.max(), 1)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            (bar.get_x() + bar.get_width() / 2, h + ymax * 0.012),
            ha="center",
            va="bottom",
            fontsize=textprops,
            color="black",
        )

    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}\ Stars}}$ (Total: {total})",
            fontsize=title_size,
            loc="left",
            pad=18,
        )
    else:
        ax.set_title(
            rf"$\bf{{Stars\ Distribution}}$ (Total: {total})",
            fontsize=title_size,
            loc="center",
            pad=18,
        )

    ax.set_xlabel("Stars (bucket)", fontsize=label_size)
    if not hide_ylabel:
        ax.set_ylabel("Number of repositories", fontsize=label_size)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=textprops)
    ax.tick_params(axis="y", labelsize=textprops)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, ymax * 1.08)
