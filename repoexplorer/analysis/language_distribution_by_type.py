#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl

from repoexplorer.analysis.type_colors import TYPE_ORDER, type_color_scale, type_sort_key


def plot_language_distribution_by_type_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
    top_n=8,
):
    """Altair stacked bar chart: language distribution by project type."""
    width = "container"
    height = "container"

    if (
        filtered_data is None
        or filtered_data.is_empty()
        or "language" not in filtered_data.columns
        or "type_prediction_gpt_5_mini" not in filtered_data.columns
    ):
        return (
            alt.Chart(pl.DataFrame({"Language": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Language Distribution")
        )

    total_repositories = filtered_data.height
    type_col = "type_prediction_gpt_5_mini"

    # Normalize language — applied to BOTH the full data and the type-filtered data
    def _normalize_lang(df):
        return df.with_columns(
            pl.when(pl.col("language").is_null()).then(pl.lit("None"))
            .when(pl.col("language") == "other").then(pl.lit("Other"))
            .when(pl.col("language") == "Jupyter Notebook").then(pl.lit("Jupyter"))
            .otherwise(pl.col("language"))
            .alias("language")
        )

    # Counts from the FULL data (before type filter) — used for % labels to match the pie
    full_lang_counts = (
        _normalize_lang(filtered_data)
        .group_by("language")
        .agg(pl.len().cast(pl.Int64).alias("AllCount"))
    )
    # Use total repos (same denominator as the pie's total_languages = filtered_data.height)
    pct_denom = total_repositories if total_repositories > 0 else 1

    # Type-filtered data for the pivot (stacked bars)
    data = _normalize_lang(filtered_data).filter(pl.col(type_col).is_not_null())

    # Cross-tab: language × project_type counts via pivot
    pivot = (
        data
        .group_by(["language", type_col])
        .agg(pl.len().alias("_c"))
        .pivot(on=type_col, index="language", values="_c", aggregate_function="sum")
        .fill_null(0)
    )
    type_cols = [c for c in pivot.columns if c != "language"]
    # Cast count columns to Int64 for consistent arithmetic downstream
    pivot = pivot.with_columns([pl.col(c).cast(pl.Int64) for c in type_cols])

    if not type_cols:
        return (
            alt.Chart(pl.DataFrame({"Language": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="Language Distribution")
        )

    # Compute per-language totals; take top N (same selection as the pie chart)
    lang_totals = (
        pivot
        .with_columns(pl.sum_horizontal([pl.col(c) for c in type_cols]).alias("_t"))
        .select(["language", "_t"])
    )
    major_langs = (
        lang_totals
        .filter(pl.col("language") != "None")
        .sort("_t", descending=True)
        .head(top_n)
        ["language"]
        .to_list()
    )

    major_pivot = pivot.filter(pl.col("language").is_in(major_langs))
    minor_pivot = pivot.filter(~pl.col("language").is_in(major_langs))
    minor_lang_names = minor_pivot["language"].to_list()

    if minor_pivot.height > 0:
        other_sums = minor_pivot.select(type_cols).sum()
        other_row = other_sums.with_columns(pl.lit("Other").alias("language")).select(pivot.columns)
        major_pivot = pl.concat([major_pivot, other_row])

    # Sort ascending by row total
    major_pivot = (
        major_pivot
        .with_columns(pl.sum_horizontal([pl.col(c) for c in type_cols]).alias("_rt"))
        .sort("_rt")
        .drop("_rt")
    )

    # "Project Type" totals row: total count of each type across all filtered data
    type_totals = (
        filtered_data.filter(pl.col(type_col).is_not_null())
        .group_by(type_col)
        .agg(pl.len().alias("_c"))
    )
    pt_dict: dict = {"language": ["Project Type"]}
    for col in type_cols:
        pt_dict[col] = [0]
    for row in type_totals.iter_rows(named=True):
        col_name = row[type_col]
        if col_name in pt_dict:
            pt_dict[col_name] = [int(row["_c"])]
    pt_row = pl.DataFrame(pt_dict).with_columns([pl.col(c).cast(pl.Int64) for c in type_cols])
    major_pivot = major_pivot.with_columns([pl.col(c).cast(pl.Int64) for c in type_cols])

    full_pivot = pl.concat([major_pivot, pt_row])
    language_order = full_pivot["language"].to_list()

    category_list = type_cols
    sorted_category_list = [t for t in TYPE_ORDER if t in category_list] + [t for t in category_list if t not in TYPE_ORDER]
    color_scale = type_color_scale(sorted_category_list)

    # Long format for stacked bars
    long_df = (
        full_pivot
        .unpivot(index="language", variable_name="ProjectType", value_name="Count")
        .filter(pl.col("Count") > 0)
        .rename({"language": "Language"})
        .with_columns(
            pl.col("ProjectType")
            .map_elements(type_sort_key, return_dtype=pl.Int64)
            .alias("_stack_order")
        )
    )

    # % labels from FULL data counts (same numerator as the pie chart)
    # "Other" = sum of minor language counts; "Project Type" = repos with type prediction
    other_all_count = int(
        full_lang_counts.filter(pl.col("language").is_in(minor_lang_names))["AllCount"].sum()
    ) if minor_lang_names else 0
    pt_all_count = int(filtered_data.filter(pl.col(type_col).is_not_null()).height)

    allcount_lookup = (
        full_lang_counts
        .filter(pl.col("language").is_in(major_langs))
        .rename({"language": "Language", "AllCount": "Total"})
    )
    extra_rows = pl.DataFrame({
        "Language": (["Other"] if minor_lang_names else []) + ["Project Type"],
        "Total": ([other_all_count] if minor_lang_names else []) + [pt_all_count],
    })
    allcount_df = pl.concat([allcount_lookup, extra_rows.with_columns(pl.col("Total").cast(pl.Int64))])

    pt_pct_label = f"{pt_all_count / total_repositories * 100:.1f}%"

    totals_df = (
        pl.DataFrame({"Language": language_order})
        .join(allcount_df, on="Language", how="left")
        .with_columns(pl.col("Total").fill_null(0))
        .with_columns(
            pl.when(pl.col("Language") == "Project Type")
            .then(pl.lit(pt_pct_label))
            .otherwise(
                (pl.col("Total").cast(pl.Float64) / pct_denom * 100)
                .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            )
            .alias("PercentLabel")
        )
    )
    # y_max still based on pivot counts (actual bar heights)
    pivot_totals = (
        full_pivot
        .with_columns(pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in type_cols]).alias("_bar_total"))
        ["_bar_total"]
    )
    y_max_val = pivot_totals.max()
    y_max = float(y_max_val * 1.12) if y_max_val is not None else 1.0
    totals_df = totals_df.with_columns(pl.lit(y_max).alias("FullHeight"))

    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("Language:N", sort=language_order, title="Language", axis=alt.Axis(labelAngle=-45, labelFontSize=label_size)),
            y=alt.Y("Count:Q", stack="zero", title="Repository Count", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(grid=True, labelFontSize=label_size)),
            color=alt.Color("ProjectType:N", scale=color_scale, sort=sorted_category_list, legend=alt.Legend(title="Project Type", orient="top-left", labelFontSize=label_size, titleFontSize=label_size)),
            order=alt.Order("_stack_order:Q", sort="ascending"),
        )
    )

    labels = (
        alt.Chart(totals_df)
        .mark_text(align="center", baseline="bottom", dy=-4, color="black", fontSize=textprops)
        .encode(x=alt.X("Language:N", sort=language_order), y=alt.Y("Total:Q"), text="PercentLabel:N")
    )

    hit_area = (
        alt.Chart(totals_df)
        .mark_bar(opacity=0)
        .encode(
            x=alt.X("Language:N", sort=language_order),
            y=alt.Y("FullHeight:Q", scale=alt.Scale(domain=[0, y_max])),
            tooltip=[alt.Tooltip("Language:N", title="Language"), alt.Tooltip("Total:Q", title="Count"), alt.Tooltip("PercentLabel:N", title="Share")],
        )
    )

    title = f"Language Distribution by Type (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels + hit_area)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
