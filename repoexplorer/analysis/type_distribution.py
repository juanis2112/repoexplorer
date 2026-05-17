#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex
import pandas as pd


def plot_type_distribution(
    filtered_data, acronym, ax=None, color_map=None, 
    title_prefix=None, label_size=25, title_size=24, textprops=15):
    """
    Plots a pie chart representing the distribution of repository types based on GPT-predicted categories.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        The filtered dataset containing the 'type_prediction_gpt_5_mini' column.

    acronym : str
        Acronym for the institution or group being plotted.

    ax : matplotlib.axes.Axes, optional
        An existing matplotlib axis to plot on. If None, creates a new figure.

    color_map : dict, optional
        A dictionary mapping category labels to colors.

    title_prefix : str, optional
        Text to prefix the plot title with.

    Returns
    -------
    None
        This function generates a pie chart but does not return any values.
    """
    # Ignore rows where type is "error" (case-insensitive)
    type_col = "type_prediction_gpt_5_mini"
    if type_col not in filtered_data.columns:
        return
    plot_data = filtered_data[
        filtered_data[type_col].astype(str).str.strip().str.lower() != "error"
    ]
    total_repositories = len(plot_data)
    category_counts = plot_data[type_col].value_counts()

    if category_counts.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        return

    labels = category_counts.index.tolist()

    cmap = cm.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(labels)}
    colors = [category_colors[cat] for cat in labels]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        category_counts,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': textprops}
    )

    for text in texts:
        text.set_fontsize(label_size)
    for autotext in autotexts:
        autotext.set_fontsize(textprops)
        
    for i, text in enumerate(texts):
        label = text.get_text().strip().upper()
    
        if label == "DOCS":
            # Move label lower
            x, y = text.get_position()
            text.set_position((x, y - 0.1))
    
            # Move percentage text slightly lower too
            x_pct, y_pct = autotexts[i].get_position()
            autotexts[i].set_position((x_pct, y_pct - 0.08))


    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{Project\ Type\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )


def plot_type_distribution_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
):
    """
    Altair version of the project type distribution pie chart for on-screen
    rendering. Mirrors the matplotlib version in `plot_type_distribution`.
    """
    width = "container"
    height = "container"
    type_col = "type_prediction_gpt_5_mini"

    if (
        filtered_data is None
        or filtered_data.empty
        or type_col not in filtered_data.columns
    ):
        return (
            alt.Chart(pd.DataFrame({"Category": [], "Count": []}))
            .mark_arc()
            .properties(width=width, height=height, title="Project Type Distribution")
        )

    plot_data = filtered_data[
        filtered_data[type_col].astype(str).str.strip().str.lower() != "error"
    ]
    total_repositories = len(plot_data)
    category_counts = plot_data[type_col].value_counts()

    if category_counts.empty:
        return (
            alt.Chart(pd.DataFrame({"Category": [], "Count": []}))
            .mark_arc()
            .properties(width=width, height=height, title="Project Type Distribution")
        )

    labels = category_counts.index.tolist()
    cmap = cm.get_cmap("tab20")
    palette = [to_hex(cmap(i)) for i in range(len(labels))]
    color_scale = alt.Scale(domain=labels, range=palette)

    plot_df = pd.DataFrame(
        {
            "Category": labels,
            "Count": [int(category_counts[c]) for c in labels],
        }
    )
    plot_df["PercentLabel"] = plot_df["Count"].apply(
        lambda c: f"{(c / total_repositories) * 100:.1f}%"
    )

    tooltip = [
        alt.Tooltip("Category:N", title="Category"),
        alt.Tooltip("Count:Q", title="Count"),
        alt.Tooltip("PercentLabel:N", title="Share"),
    ]

    base = alt.Chart(plot_df).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color(
            "Category:N",
            scale=color_scale,
            legend=alt.Legend(
                title=None,
                labelFontSize=label_size,
                orient="top-left",
            ),
        ),
        tooltip=tooltip,
    )

    # Scale radii with the container so the pie reacts to the card size.
    outer_radius_expr = "min(width, height) / 2 - 10"
    text_radius_expr = "min(width, height) / 2 + 10"

    arcs = base.mark_arc(outerRadius=alt.expr(outer_radius_expr))
    pct_text = base.mark_text(
        radius=alt.expr(text_radius_expr),
        fontSize=textprops,
    ).encode(
        text="PercentLabel:N",
        color=alt.value("black"),
    )

    title = f"Project Type Distribution (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (arcs + pct_text)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_view(stroke=None)
    )
