#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl
import random

_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_FEATURE_DISPLAY_NAMES = {
    "description": "Description",
    "readme": "README",
    "license": "License",
    "code_of_conduct_file": "Code of Conduct",
    "contributing": "Contributing Guide",
    "security_policy": "Security Policy",
    "issue_templates": "Issue Templates",
    "pull_request_template": "PR Template",
}


def plot_feature_counts_altair(
    data,
    features,
    acronym="",
    label_size=8,
    title_size=12,
    textprops=8,
):
    """Altair bar chart: count of repos with each community file feature."""
    width = "container"
    height = "container"

    if data is None or data.is_empty():
        return (
            alt.Chart(pl.DataFrame({"Feature": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Community Files Presence")
        )

    total_repositories = data.height

    # Count non-nulls per feature, map to display names
    feature_counts: dict[str, int] = {}
    for f in features:
        if f in data.columns:
            display = _FEATURE_DISPLAY_NAMES.get(f, f)
            feature_counts[display] = int(data[f].is_not_null().sum())

    if not feature_counts:
        return (
            alt.Chart(pl.DataFrame({"Feature": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Community Files Presence")
        )

    # Sort ascending
    order = sorted(feature_counts, key=lambda k: feature_counts[k])

    palette = list(_TAB10)
    random.seed("39")
    random.shuffle(palette)
    palette = palette[: len(order)]
    color_scale = alt.Scale(domain=order, range=palette)

    plot_df = pl.DataFrame({
        "Feature": order,
        "Count": [feature_counts[f] for f in order],
    })
    plot_df = plot_df.with_columns(
        (pl.col("Count").cast(pl.Float64) / total_repositories * 100)
        .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
        .alias("PercentLabel")
    )
    y_max_val = plot_df["Count"].max()
    y_max = float(y_max_val * 1.12) if y_max_val is not None else 1.0
    plot_df = plot_df.with_columns(pl.lit(y_max).alias("FullHeight"))

    tooltip = [
        alt.Tooltip("Feature:N", title="Feature"),
        alt.Tooltip("Count:Q", title="Count"),
        alt.Tooltip("PercentLabel:N", title="Share"),
    ]

    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Feature:N", sort=order, title="Feature", axis=alt.Axis(labelAngle=-45, labelFontSize=label_size)),
            y=alt.Y("Count:Q", title="Repository Count", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(grid=True, labelFontSize=label_size)),
            color=alt.Color("Feature:N", scale=color_scale, legend=None),
        )
    )

    hit_area = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0)
        .encode(x=alt.X("Feature:N", sort=order), y=alt.Y("FullHeight:Q", scale=alt.Scale(domain=[0, y_max])), tooltip=tooltip)
    )

    labels = (
        alt.Chart(plot_df)
        .mark_text(align="center", baseline="bottom", dy=-4, color="black", fontSize=textprops)
        .encode(x=alt.X("Feature:N", sort=order), y=alt.Y("Count:Q"), text="PercentLabel:N")
    )

    title = f"Community Files Presence (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels + hit_area)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
