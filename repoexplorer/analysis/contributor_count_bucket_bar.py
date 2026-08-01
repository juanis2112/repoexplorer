#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bucket bar chart for ``contributor_count`` (Sustainability tab).
Distinct from ``contributors_distribution_bar.py`` (Impact tab star-style buckets).
"""

import polars as pl
import altair as alt


_BUCKET_LABELS = ["0-2", "3-10", "10-50", "50-100", "100+"]
_VALUE_COL = "contributor_count"


def _bucket_expr() -> pl.Expr:
    """
    Map contributor counts to buckets. NaN / negative -> null.
    Ranges: [0,2], [3,10], (10,50], (50,100], >100 so 10 is only in 3-10.
    """
    v = pl.col("_v")
    return (
        pl.when(v.is_not_null() & (v >= 0) & (v <= 2)).then(pl.lit(_BUCKET_LABELS[0]))
        .when(v.is_not_null() & (v >= 3) & (v <= 10)).then(pl.lit(_BUCKET_LABELS[1]))
        .when(v.is_not_null() & (v > 10) & (v <= 50)).then(pl.lit(_BUCKET_LABELS[2]))
        .when(v.is_not_null() & (v > 50) & (v <= 100)).then(pl.lit(_BUCKET_LABELS[3]))
        .when(v.is_not_null() & (v > 100)).then(pl.lit(_BUCKET_LABELS[4]))
        .otherwise(pl.lit(None))
        .alias("bucket")
    )


def plot_contributor_count_bucket_bar_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=9,
):
    """Altair bar chart: repositories per contributor-count bucket."""
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
            .properties(width=width, height=height, title="Contributor Count Distribution")
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
            x=alt.X("bucket:N", sort=_BUCKET_LABELS, title="Contributors (bucket)", axis=alt.Axis(labelAngle=-30, labelFontSize=label_size)),
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

    title = f"Contributor Count Distribution (Total: {total})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
