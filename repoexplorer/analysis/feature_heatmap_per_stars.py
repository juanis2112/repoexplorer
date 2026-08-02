#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl


_FEATURE_DISPLAY_NAMES = {
    "description": "Description",
    "readme": "README",
    "license": "License",
    "code_of_conduct_file": "Code of Conduct",
    "contributing": "Contributing Guide",
    "security_policy": "Security Policy",
    "issue_templates": "Issue Templates",
    "pull_request_template": "PR Template",
    "type_prediction_gpt_5_mini": "Project Type",
}


def plot_feature_heatmap_by_star_bucket_altair(
    df,
    features,
    star_col="stargazers_count",
    label_size=10,
    title_size=12,
    annotations_size=9,
):
    """Altair heatmap: % of repos with each feature, grouped by star bucket."""
    width = "container"
    height = "container"

    star_buckets = ["0–10", "11–50", "51–100", "101–200", ">200"]

    _EMPTY = pl.DataFrame({
        "Feature": pl.Series([], dtype=pl.Utf8),
        "Bucket": pl.Series([], dtype=pl.Utf8),
        "Percentage": pl.Series([], dtype=pl.Float64),
    })

    if df is None or df.is_empty() or star_col not in df.columns:
        return (
            alt.Chart(_EMPTY)
            .mark_rect()
            .properties(width=width, height=height, title="Community Files by # Stars")
        )

    total_repositories = df.height

    # Assign star buckets using Polars when/then
    star_expr = pl.col(star_col).cast(pl.Float64, strict=False)
    data = df.with_columns(
        pl.when(star_expr.is_null()).then(pl.lit(None))
        .when(star_expr <= 10).then(pl.lit("0–10"))
        .when(star_expr <= 50).then(pl.lit("11–50"))
        .when(star_expr <= 100).then(pl.lit("51–100"))
        .when(star_expr <= 200).then(pl.lit("101–200"))
        .otherwise(pl.lit(">200"))
        .alias("star_bucket")
    )

    rows = []
    feature_order = []

    for feature in features:
        if feature not in data.columns:
            continue
        display = _FEATURE_DISPLAY_NAMES.get(feature, feature)
        feature_order.append(display)
        for bucket in star_buckets:
            subset = data.filter(pl.col("star_bucket") == bucket)
            total_b = subset.height
            count = int(subset[feature].is_not_null().sum()) if total_b > 0 else 0
            pct = (count / total_b * 100) if total_b > 0 else 0.0
            rows.append({"Feature": display, "Bucket": bucket, "Percentage": round(pct, 1)})

    if not rows:
        return (
            alt.Chart(_EMPTY)
            .mark_rect()
            .properties(width=width, height=height, title="Community Files by # Stars")
        )

    # Append "Average" row per bucket
    for bucket in star_buckets:
        bucket_rows = [r for r in rows if r["Bucket"] == bucket]
        avg = sum(r["Percentage"] for r in bucket_rows) / len(bucket_rows) if bucket_rows else 0.0
        rows.append({"Feature": "Average", "Bucket": bucket, "Percentage": round(avg, 1)})
    feature_order.append("Average")

    long_df = pl.DataFrame(rows).with_columns(
        pl.col("Percentage")
        .map_elements(lambda p: f"{p:.0f}", return_dtype=pl.Utf8)
        .alias("Label")
    )

    rects = (
        alt.Chart(long_df)
        .mark_rect()
        .encode(
            x=alt.X("Bucket:N", sort=star_buckets, title="# Star Bucket", scale=alt.Scale(paddingOuter=0, paddingInner=0.05), axis=alt.Axis(labelFontSize=label_size, labelAngle=-30)),
            y=alt.Y("Feature:N", sort=feature_order, title="Community File", scale=alt.Scale(paddingOuter=0, paddingInner=0.05), axis=alt.Axis(labelFontSize=label_size)),
            color=alt.Color("Percentage:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100]), legend=alt.Legend(title="% with feature", labelFontSize=label_size, titleFontSize=label_size, titleOrient="right")),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Bucket:N", title="Star Bucket"), alt.Tooltip("Percentage:Q", title="%", format=".1f")],
        )
    )

    texts = (
        alt.Chart(long_df)
        .mark_text(fontSize=annotations_size, color="black")
        .encode(x=alt.X("Bucket:N", sort=star_buckets), y=alt.Y("Feature:N", sort=feature_order), text="Label:N")
    )

    title = f"Community Files by # Stars (Total: {total_repositories})"

    return (
        (rects + texts)
        .properties(width=width, height=alt.Step(35), title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_legend(gradientLength=alt.ExprRef("height"))
        .configure_view(stroke=None)
    )
