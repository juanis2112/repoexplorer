#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import pandas as pd


def prepare_pie_label_data(plot_df, count_col="Count"):
    """Preserve slice order so arcs and labels stack identically."""
    df = plot_df.copy()
    df["stack_order"] = range(len(df))
    return df


def pie_arc_layer(plot_df, outer_radius_expr, color_field, color_scale, legend, tooltip):
    return (
        alt.Chart(plot_df)
        .mark_arc(outerRadius=alt.expr(outer_radius_expr))
        .encode(
            theta=alt.Theta("Count:Q", stack=True),
            order=alt.Order("stack_order:Q"),
            color=alt.Color(color_field, scale=color_scale, legend=legend),
            tooltip=tooltip,
        )
    )


def pie_pct_label_layer(
    plot_df,
    text_radius_expr,
    textprops,
    label_field="PercentLabel",
    min_label_pct=0.03,
):
    total = float(plot_df["Count"].sum())
    theta_scale = alt.Scale(domain=[0, total]) if total > 0 else alt.Undefined

    return (
        alt.Chart(plot_df)
        .transform_stack(
            stack="Count",
            groupby=[],
            sort=[alt.SortField("stack_order", order="ascending")],
            as_=["theta_start", "theta_end"],
        )
        .transform_joinaggregate(total_count="sum(Count)")
        .transform_calculate(
            theta_mid="(datum.theta_start + datum.theta_end) / 2",
            label_text=(
                f"datum.Count / datum.total_count > {min_label_pct} "
                f"? datum.{label_field} : ''"
            ),
        )
        .mark_text(
            radius=alt.expr(text_radius_expr),
            fontSize=textprops,
            fontWeight="bold",
        )
        .encode(
            theta=alt.Theta("theta_mid:Q", stack=None, scale=theta_scale),
            text="label_text:N",
            color=alt.value("black"),
        )
    )
