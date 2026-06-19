#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt


_BUCKET_LABELS = ["0-10", "11-100", "101-1000", "1K-10K", "10K+"]
_VALUE_COL = "forks_count"


def _values_to_bucket_labels(values: pd.Series) -> pd.Series:
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


def plot_forks_distribution_bar(
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
    Bar chart: count of repositories per forks bucket (0-10, 11-100, ...).
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
    cmap = plt.colormaps["tab10"]
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
            rf"$\bf{{{title_prefix}\ {acronym}\ Forks}}$ (Total: {total})",
            fontsize=title_size,
            loc="left",
            pad=18,
        )
    else:
        ax.set_title(
            rf"$\bf{{Forks\ Distribution}}$ (Total: {total})",
            fontsize=title_size,
            loc="center",
            pad=18,
        )

    ax.set_xlabel("Forks (bucket)", fontsize=label_size)
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


def plot_forks_distribution_bar_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=9,
):
    """Altair bar chart: count of repositories per forks bucket."""
    width = "container"
    height = "container"

    if (
        filtered_data is None
        or (hasattr(filtered_data, "empty") and filtered_data.empty)
        or _VALUE_COL not in filtered_data.columns
    ):
        return (
            alt.Chart(pd.DataFrame({"bucket": [], "Count": []}))
            .mark_bar()
            .properties(width=width, height=height, title="Forks Distribution")
        )

    total = len(filtered_data)
    buckets = _values_to_bucket_labels(filtered_data[_VALUE_COL])
    counts = (
        buckets.dropna()
        .value_counts()
        .reindex(_BUCKET_LABELS, fill_value=0)
        .reset_index()
    )
    counts.columns = ["bucket", "Count"]
    counts["Label"] = counts["Count"].apply(lambda c: f"{c / total * 100:.1f}%")
    y_max = int(counts["Count"].max() * 1.15) + 1

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_scale = alt.Scale(domain=_BUCKET_LABELS, range=colors)

    bars = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X(
                "bucket:N",
                sort=_BUCKET_LABELS,
                title="Forks (bucket)",
                axis=alt.Axis(labelAngle=-30, labelFontSize=label_size),
            ),
            y=alt.Y(
                "Count:Q",
                title="Number of repositories",
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(grid=True, labelFontSize=label_size),
            ),
            color=alt.Color("bucket:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("bucket:N", title="Bucket"),
                alt.Tooltip("Count:Q", title="Count"),
            ],
        )
    )

    labels = (
        alt.Chart(counts)
        .mark_text(
            align="center", baseline="bottom", dy=-4, fontSize=textprops, color="black"
        )
        .encode(
            x=alt.X("bucket:N", sort=_BUCKET_LABELS),
            y=alt.Y("Count:Q"),
            text="Label:N",
        )
    )

    title = f"Forks Distribution (Total: {total})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
