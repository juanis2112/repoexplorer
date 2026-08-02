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

_EMPTY_DF = pl.DataFrame({"Language": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)})

LANGUAGE_LABEL_MAP = {
    "Jupyter Notebook": "Jupyter",
}


def plot_language_distribution_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
    top_n=8,
):
    """Altair pie chart: language distribution (major languages + 'Other' bucket)."""
    width = "container"
    height = "container"

    if (
        filtered_data is None
        or filtered_data.is_empty()
        or "language" not in filtered_data.columns
    ):
        return (
            alt.Chart(_EMPTY_DF)
            .mark_arc()
            .properties(width=width, height=height, title="Language Distribution")
        )

    total_repositories = filtered_data.height

    # Normalize language labels
    data = filtered_data.with_columns(
        pl.when(pl.col("language").is_null()).then(pl.lit("None"))
        .when(pl.col("language") == "Jupyter Notebook").then(pl.lit("Jupyter"))
        .otherwise(pl.col("language"))
        .alias("language")
    )

    counts = (
        data
        .group_by("language")
        .agg(pl.len().cast(pl.Int64).alias("Count"))
        .sort("Count", descending=True)
    )

    total_languages = counts["Count"].sum()

    if counts.is_empty() or total_languages == 0:
        return (
            alt.Chart(_EMPTY_DF)
            .mark_arc()
            .properties(width=width, height=height, title="Language Distribution")
        )

    # Take the top N languages by count (excluding "None"); rest go to "Other"
    langs_ranked = (
        counts
        .filter(pl.col("language") != "None")
        .sort("Count", descending=True)
    )
    top_names = set(langs_ranked.head(top_n)["language"].to_list())

    major = counts.filter(pl.col("language").is_in(top_names))
    minor = counts.filter(
        ~pl.col("language").is_in(top_names) & (pl.col("language") != "None")
    )

    if minor.height > 0:
        other_count = int(minor["Count"].sum())
        other_row = pl.DataFrame({"language": ["Other"], "Count": pl.Series([other_count], dtype=pl.Int64)})
        lang_grouped = pl.concat([major, other_row])
    else:
        lang_grouped = major

    labels = lang_grouped["language"].to_list()
    palette = [_TAB20[i % 20] for i in range(len(labels))]
    color_scale = alt.Scale(domain=labels, range=palette)

    plot_df = (
        lang_grouped
        .rename({"language": "Language"})
        .with_columns(
            (pl.col("Count").cast(pl.Float64) / total_languages * 100)
            .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            .alias("PercentLabel")
        )
    )
    plot_df = prepare_pie_label_data(plot_df)

    tooltip = [
        alt.Tooltip("Language:N", title="Language"),
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

    arcs = pie_arc_layer(plot_df, outer_radius_expr, "Language:N", color_scale, legend, tooltip)
    pct_text = pie_pct_label_layer(plot_df, text_radius_expr, textprops)

    title = f"Language Distribution (Total: {total_repositories})"
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
