#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl

from repoexplorer.analysis.altair_pie_helpers import (
    pie_arc_layer,
    pie_pct_label_layer,
    prepare_pie_label_data,
)

_TAB20 = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]

_EMPTY_DF = pl.DataFrame({"License": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)})


def plot_license_distribution_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
    top_n=8,
):
    """Altair pie chart: license distribution (major licenses + 'Other' bucket)."""
    width = "container"
    height = "container"

    if (
        filtered_data is None
        or filtered_data.is_empty()
        or "license" not in filtered_data.columns
    ):
        return (
            alt.Chart(_EMPTY_DF)
            .mark_arc()
            .properties(width=width, height=height, title="License Distribution")
        )

    total_repositories = filtered_data.height

    # Normalize: null → "None", lowercase "other" → "Other"
    data = filtered_data.with_columns(
        pl.when(pl.col("license").is_null()).then(pl.lit("None"))
        .when(pl.col("license") == "other").then(pl.lit("Other"))
        .otherwise(pl.col("license"))
        .alias("license")
    )

    counts = (
        data
        .filter(pl.col("license").is_not_null())
        .group_by("license")
        .agg(pl.len().cast(pl.Int64).alias("Count"))
        .sort("Count", descending=True)
    )

    total_licenses = counts["Count"].sum()

    if counts.is_empty() or total_licenses == 0:
        return (
            alt.Chart(_EMPTY_DF)
            .mark_arc()
            .properties(width=width, height=height, title="License Distribution")
        )

    # Take top N licenses by count (includes "None"); rest go to "Other"
    top_names = set(counts.sort("Count", descending=True).head(top_n)["license"].to_list())

    major = counts.filter(pl.col("license").is_in(top_names))
    minor = counts.filter(~pl.col("license").is_in(top_names))

    if minor.height > 0:
        other_count = int(minor["Count"].sum())
        # If "Other" already exists in major (from normalized "other"), add to it
        if "Other" in major["license"].to_list():
            major = major.with_columns(
                pl.when(pl.col("license") == "Other")
                .then(pl.col("Count") + pl.lit(other_count, dtype=pl.Int64))
                .otherwise(pl.col("Count"))
                .alias("Count")
            )
        else:
            other_row = pl.DataFrame({"license": ["Other"], "Count": pl.Series([other_count], dtype=pl.Int64)})
            major = pl.concat([major, other_row])

    lic_grouped = major
    labels = lic_grouped["license"].to_list()
    palette = [_TAB20[i % 20] for i in range(len(labels))]
    color_scale = alt.Scale(domain=labels, range=palette)

    plot_df = (
        lic_grouped
        .rename({"license": "License"})
        .with_columns(
            (pl.col("Count").cast(pl.Float64) / total_licenses * 100)
            .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            .alias("PercentLabel")
        )
    )
    plot_df = prepare_pie_label_data(plot_df)

    tooltip = [
        alt.Tooltip("License:N", title="License"),
        alt.Tooltip("Count:Q", title="Count"),
        alt.Tooltip("PercentLabel:N", title="Share"),
    ]

    outer_radius_expr = "min(width, height) * 0.38"
    text_radius_expr = "min(width, height) * 0.22"

    legend = alt.Legend(
        title=None,
        labelFontSize=label_size,
        orient="bottom-right",
        offset=0,
        padding=0,
    )

    arcs = pie_arc_layer(plot_df, outer_radius_expr, "License:N", color_scale, legend, tooltip)
    pct_text = pie_pct_label_layer(plot_df, text_radius_expr, textprops)

    title = f"License Distribution (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (arcs + pct_text)
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=title_size, anchor="middle"),
        )
        .configure(padding={"right": 0, "bottom": 0})
        .configure_view(stroke=None)
    )
