#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import polars as pl

from repoexplorer.analysis.type_colors import TYPE_ORDER, type_color_scale, type_sort_key


def plot_license_distribution_by_type_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
    top_n=8,
):
    """Altair stacked bar chart: license distribution by project type."""
    width = "container"
    height = "container"

    if (
        filtered_data is None
        or filtered_data.is_empty()
        or "license" not in filtered_data.columns
        or "type_prediction_gpt_5_mini" not in filtered_data.columns
    ):
        return (
            alt.Chart(pl.DataFrame({"License": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="License Distribution")
        )

    total_repositories = filtered_data.height
    type_col = "type_prediction_gpt_5_mini"

    # Use total_repositories as denominator — same as the pie chart, which counts
    # null licenses as "None" and includes them in the total.
    pct_denom = total_repositories if total_repositories > 0 else 1

    def _normalize_lic(df):
        return df.with_columns(
            pl.when(pl.col("license").is_null()).then(pl.lit("None"))
            .when(pl.col("license") == "other").then(pl.lit("Other"))
            .otherwise(pl.col("license"))
            .alias("license")
        )

    # Counts from FULL data (before type filter) — used for % labels to match the pie
    full_lic_counts = (
        _normalize_lic(filtered_data)
        .group_by("license")
        .agg(pl.len().cast(pl.Int64).alias("AllCount"))
    )

    # Normalize license values; filter out null types
    data = _normalize_lic(filtered_data).filter(pl.col(type_col).is_not_null())

    # Cross-tab: license × project_type counts via pivot
    pivot = (
        data
        .group_by(["license", type_col])
        .agg(pl.len().alias("_c"))
        .pivot(on=type_col, index="license", values="_c", aggregate_function="sum")
        .fill_null(0)
    )
    type_cols = [c for c in pivot.columns if c != "license"]
    # Cast count columns to Int64 for consistent arithmetic downstream
    pivot = pivot.with_columns([pl.col(c).cast(pl.Int64) for c in type_cols])

    if not type_cols:
        return (
            alt.Chart(pl.DataFrame({"License": pl.Series([], dtype=pl.Utf8), "Count": pl.Series([], dtype=pl.Int64)}))
            .mark_bar()
            .properties(width=width, height=height, title="License Distribution")
        )

    # Compute per-license totals for major/minor split
    lic_totals = (
        pivot
        .with_columns(pl.sum_horizontal([pl.col(c) for c in type_cols]).alias("_t"))
        .select(["license", "_t"])
    )
    major_lics = (
        lic_totals
        .sort("_t", descending=True)
        .head(top_n)
        ["license"]
        .to_list()
    )

    major_pivot = pivot.filter(pl.col("license").is_in(major_lics))
    minor_pivot = pivot.filter(~pl.col("license").is_in(major_lics))

    if minor_pivot.height > 0:
        other_sums = minor_pivot.select(type_cols).sum()
        if "Other" in major_pivot["license"].to_list():
            # Merge into existing "Other" row
            other_counts = other_sums.row(0)
            other_exprs = [
                pl.when(pl.col("license") == "Other")
                .then(pl.col(c) + int(other_counts[i]))
                .otherwise(pl.col(c))
                .alias(c)
                for i, c in enumerate(type_cols)
            ]
            major_pivot = major_pivot.with_columns(other_exprs)
        else:
            other_row = other_sums.with_columns(pl.lit("Other").alias("license")).select(pivot.columns)
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
    pt_dict: dict = {"license": ["Project Type"]}
    for col in type_cols:
        pt_dict[col] = [0]
    for row in type_totals.iter_rows(named=True):
        col_name = row[type_col]
        if col_name in pt_dict:
            pt_dict[col_name] = [int(row["_c"])]
    pt_row = pl.DataFrame(pt_dict).with_columns([pl.col(c).cast(pl.Int64) for c in type_cols])
    major_pivot = major_pivot.with_columns([pl.col(c).cast(pl.Int64) for c in type_cols])

    full_pivot = pl.concat([major_pivot, pt_row])
    license_order = full_pivot["license"].to_list()

    category_list = type_cols
    sorted_category_list = [t for t in TYPE_ORDER if t in category_list] + [t for t in category_list if t not in TYPE_ORDER]
    color_scale = type_color_scale(sorted_category_list)

    # Long format for stacked bars
    long_df = (
        full_pivot
        .unpivot(index="license", variable_name="ProjectType", value_name="Count")
        .filter(pl.col("Count") > 0)
        .rename({"license": "License"})
        .with_columns(
            pl.col("ProjectType")
            .map_elements(type_sort_key, return_dtype=pl.Int64)
            .alias("_stack_order")
        )
    )

    # Bar totals (from pivot — used for y-positioning labels and y_max)
    bar_totals_df = (
        full_pivot
        .with_columns(pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in type_cols]).alias("BarTotal"))
        .select(["license", "BarTotal"])
        .rename({"license": "License"})
    )
    y_max_val = bar_totals_df["BarTotal"].max()
    y_max = float(y_max_val * 1.12) if y_max_val is not None else 1.0

    # Full-data counts (numerator for % text, matching the pie chart)
    minor_lic_names = minor_pivot["license"].to_list()
    other_all_count = int(
        full_lic_counts.filter(pl.col("license").is_in(minor_lic_names))["AllCount"].sum()
    ) if minor_lic_names else 0
    pt_all_count = int(filtered_data.filter(pl.col(type_col).is_not_null()).height)
    pt_pct_label = f"{pt_all_count / total_repositories * 100:.1f}%"

    allcount_lookup = (
        full_lic_counts
        .filter(pl.col("license").is_in(major_lics))
        .rename({"license": "License", "AllCount": "AllTotal"})
    )
    # If "Other" is already in major_lics (from normalized "other" entries), merge
    # the minor-license count into it rather than adding a duplicate row.
    if minor_lic_names and other_all_count > 0:
        if "Other" in allcount_lookup["License"].to_list():
            allcount_lookup = allcount_lookup.with_columns(
                pl.when(pl.col("License") == "Other")
                .then(pl.col("AllTotal") + pl.lit(other_all_count, dtype=pl.Int64))
                .otherwise(pl.col("AllTotal"))
                .alias("AllTotal")
            )
            pt_extra = pl.DataFrame({"License": ["Project Type"], "AllTotal": pl.Series([pt_all_count], dtype=pl.Int64)})
            allcount_df = pl.concat([allcount_lookup, pt_extra])
        else:
            extra_rows = pl.DataFrame({
                "License": ["Other", "Project Type"],
                "AllTotal": pl.Series([other_all_count, pt_all_count], dtype=pl.Int64),
            })
            allcount_df = pl.concat([allcount_lookup, extra_rows])
    else:
        pt_extra = pl.DataFrame({"License": ["Project Type"], "AllTotal": pl.Series([pt_all_count], dtype=pl.Int64)})
        allcount_df = pl.concat([allcount_lookup, pt_extra])

    totals_df = (
        pl.DataFrame({"License": license_order})
        .join(bar_totals_df, on="License", how="left")
        .join(allcount_df, on="License", how="left")
        .with_columns([pl.col("BarTotal").fill_null(0), pl.col("AllTotal").fill_null(0)])
        .with_columns(
            pl.when(pl.col("License") == "Project Type")
            .then(pl.lit(pt_pct_label))
            .otherwise(
                (pl.col("AllTotal").cast(pl.Float64) / pct_denom * 100)
                .map_elements(lambda c: f"{c:.1f}%", return_dtype=pl.Utf8)
            )
            .alias("PercentLabel")
        )
        .with_columns(pl.lit(y_max).alias("FullHeight"))
    )

    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("License:N", sort=license_order, title="License", axis=alt.Axis(labelAngle=-45, labelFontSize=label_size)),
            y=alt.Y("Count:Q", stack="zero", title="Repository Count", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(grid=True, labelFontSize=label_size)),
            color=alt.Color("ProjectType:N", scale=color_scale, sort=sorted_category_list, legend=alt.Legend(title="Project Type", orient="top-left", labelFontSize=label_size, titleFontSize=label_size)),
            order=alt.Order("_stack_order:Q", sort="ascending"),
        )
    )

    labels = (
        alt.Chart(totals_df)
        .mark_text(align="center", baseline="bottom", dy=-4, color="black", fontSize=textprops)
        .encode(x=alt.X("License:N", sort=license_order), y=alt.Y("BarTotal:Q", scale=alt.Scale(domain=[0, y_max])), text="PercentLabel:N")
    )

    hit_area = (
        alt.Chart(totals_df)
        .mark_bar(opacity=0)
        .encode(
            x=alt.X("License:N", sort=license_order),
            y=alt.Y("FullHeight:Q", scale=alt.Scale(domain=[0, y_max])),
            tooltip=[alt.Tooltip("License:N", title="License"), alt.Tooltip("AllTotal:Q", title="Count"), alt.Tooltip("PercentLabel:N", title="Share")],
        )
    )

    title = f"License Distribution by Type (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels + hit_area)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
