#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polars as pl
import altair as alt


_BUCKET_LABELS = ["0-10", "11-100", "101-1000", "1K-10K", "10K+"]
_VALUE_COL = "forks_count"


def _bucket_expr() -> pl.Expr:
    v = pl.col("_v")
    return (
        pl.when(v.is_not_null() & (v >= 0) & (v <= 10)).then(pl.lit(_BUCKET_LABELS[0]))
        .when(v.is_not_null() & (v > 10) & (v <= 100)).then(pl.lit(_BUCKET_LABELS[1]))
        .when(v.is_not_null() & (v > 100) & (v <= 1000)).then(pl.lit(_BUCKET_LABELS[2]))
        .when(v.is_not_null() & (v > 1000) & (v <= 10000)).then(pl.lit(_BUCKET_LABELS[3]))
        .when(v.is_not_null() & (v > 10000)).then(pl.lit(_BUCKET_LABELS[4]))
        .otherwise(pl.lit(None))
        .alias("bucket")
    )


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
        or filtered_data.is_empty()
        or _VALUE_COL not in filtered_data.columns
    ):
        return (
            alt.Chart(pl.DataFrame({"bucket": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Forks Distribution")
        )

    total = filtered_data.height
    bucketed = (
        filtered_data
        .select(pl.col(_VALUE_COL).cast(pl.Float64, strict=False).alias("_v"))
        .with_columns(_bucket_expr())
        .filter(pl.col("bucket").is_not_null())
        .group_by("bucket")
        .agg(pl.len().alias("Count"))
    )
    counts = (
        pl.DataFrame({"bucket": _BUCKET_LABELS})
        .join(bucketed, on="bucket", how="left")
        .with_columns(pl.col("Count").fill_null(0))
        .with_columns(
            (pl.col("Count").cast(pl.Float64) / total * 100)
            .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            .alias("Label")
        )
    )

    y_max = int(counts["Count"].max() * 1.15) + 1
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_scale = alt.Scale(domain=_BUCKET_LABELS, range=colors)

    bars = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("bucket:N", sort=_BUCKET_LABELS, title="Forks (bucket)", axis=alt.Axis(labelAngle=-30, labelFontSize=label_size)),
            y=alt.Y("Count:Q", title="Number of repositories", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(grid=True, labelFontSize=label_size)),
            color=alt.Color("bucket:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("bucket:N", title="Bucket"), alt.Tooltip("Count:Q", title="Count")],
        )
    )
    labels = (
        alt.Chart(counts)
        .mark_text(align="center", baseline="bottom", dy=-4, fontSize=textprops, color="black")
        .encode(x=alt.X("bucket:N", sort=_BUCKET_LABELS), y=alt.Y("Count:Q"), text="Label:N")
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
