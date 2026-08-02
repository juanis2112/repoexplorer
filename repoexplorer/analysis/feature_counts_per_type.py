#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl

from repoexplorer.analysis.type_colors import TYPE_ORDER, type_color_scale, type_sort_key

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


def plot_feature_counts_per_type_altair(
    filtered_data,
    features,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=9,
):
    """Altair stacked bar chart: feature presence counts across GPT-predicted project types."""
    width = "container"
    height = "container"
    type_col = "type_prediction_gpt_5_mini"

    if (
        filtered_data is None
        or filtered_data.is_empty()
        or type_col not in filtered_data.columns
    ):
        return (
            alt.Chart(pl.DataFrame({"feature": pl.Series([], dtype=pl.Utf8), "project_type": pl.Series([], dtype=pl.Utf8), "count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Community Files")
        )

    total_repositories = filtered_data.height

    # Only rows with a known project type
    type_data = filtered_data.filter(pl.col(type_col).is_not_null())

    available_features = [f for f in features if f in type_data.columns and f != type_col]
    if not available_features:
        return (
            alt.Chart(pl.DataFrame({"feature": pl.Series([], dtype=pl.Utf8), "project_type": pl.Series([], dtype=pl.Utf8), "count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Community Files")
        )

    display_features = [_FEATURE_DISPLAY_NAMES.get(f, f) for f in available_features]

    # Count non-nulls per (type, feature) in one group_by pass
    agg_exprs = [pl.col(f).is_not_null().sum().alias(f) for f in available_features]
    type_feat_counts = type_data.group_by(type_col).agg(agg_exprs)
    raw_types = type_feat_counts[type_col].to_list()
    project_types = [t for t in TYPE_ORDER if t in raw_types] + [t for t in raw_types if t not in TYPE_ORDER]

    # Build long format and compute feature totals for ordering
    rows = []
    feature_totals: dict[str, int] = {}
    for feat_raw, feat_display in zip(available_features, display_features):
        total = 0
        for row in type_feat_counts.iter_rows(named=True):
            count = int(row[feat_raw])
            rows.append({
                "feature": feat_display,
                "project_type": row[type_col],
                "count": count,
                "_stack_order": type_sort_key(row[type_col]),
            })
            total += count
        feature_totals[feat_display] = total

    # Order features ascending by total count (smallest on left)
    order = sorted(feature_totals, key=lambda k: feature_totals[k])

    long_df = pl.DataFrame(rows)

    color_scale = type_color_scale(project_types)

    # Totals per feature for percentage labels
    totals = (
        long_df
        .group_by("feature")
        .agg(pl.col("count").sum().alias("count"))
        .with_columns(
            (pl.col("count").cast(pl.Float64) / total_repositories * 100)
            .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            .alias("pct")
        )
    )

    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("feature:N", sort=order, title="Community File", axis=alt.Axis(labelAngle=-45, labelFontSize=label_size, labelLimit=200)),
            y=alt.Y("count:Q", title="Repository Count", stack="zero", axis=alt.Axis(grid=True, labelFontSize=label_size)),
            color=alt.Color("project_type:N", scale=color_scale, sort=project_types, title="Project Type", legend=alt.Legend(labelFontSize=label_size, titleFontSize=label_size, orient="top-left")),
            order=alt.Order("_stack_order:Q", sort="ascending"),
            tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("project_type:N", title="Project Type"), alt.Tooltip("count:Q", title="Count")],
        )
    )

    labels = (
        alt.Chart(totals)
        .mark_text(align="center", baseline="bottom", dy=-4, fontSize=textprops, color="black")
        .encode(x=alt.X("feature:N", sort=order), y=alt.Y("count:Q"), text="pct:N")
    )

    title = f"Community Files (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
