#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl

from repoexplorer.analysis.altair_pie_helpers import (
    pie_arc_layer,
    pie_pct_label_layer,
    prepare_pie_label_data,
)
from repoexplorer.analysis.type_colors import type_color_scale

_EMPTY_DF = pl.DataFrame({"Category": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)})


def plot_type_distribution_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
):
    """Altair pie chart: distribution of repository types (GPT-predicted categories)."""
    width = "container"
    height = "container"
    type_col = "type_prediction_gpt_5_mini"

    if (
        filtered_data is None
        or filtered_data.is_empty()
        or type_col not in filtered_data.columns
    ):
        return (
            alt.Chart(_EMPTY_DF)
            .mark_arc()
            .properties(width=width, height=height, title="Project Type Distribution")
        )

    # Filter out "error" predictions
    plot_data = filtered_data.filter(
        pl.col(type_col).cast(pl.Utf8).str.strip_chars().str.to_lowercase() != "error"
    )
    total_repositories = plot_data.height

    counts = (
        plot_data
        .filter(pl.col(type_col).is_not_null())
        .group_by(type_col)
        .agg(pl.len().cast(pl.Int64).alias("Count"))
        .sort("Count", descending=True)
    )

    if counts.is_empty():
        return (
            alt.Chart(_EMPTY_DF)
            .mark_arc()
            .properties(width=width, height=height, title="Project Type Distribution")
        )

    labels = counts[type_col].to_list()
    color_scale = type_color_scale(labels)

    plot_df = (
        counts
        .rename({type_col: "Category"})
        .with_columns(
            (pl.col("Count").cast(pl.Float64) / total_repositories * 100)
            .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            .alias("PercentLabel")
        )
    )
    plot_df = prepare_pie_label_data(plot_df)

    tooltip = [
        alt.Tooltip("Category:N", title="Category"),
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

    arcs = pie_arc_layer(plot_df, outer_radius_expr, "Category:N", color_scale, legend, tooltip)
    pct_text = pie_pct_label_layer(plot_df, text_radius_expr, textprops)

    title = f"Project Type Distribution (Total: {total_repositories})"
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
