#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import pandas as pd
import altair as alt
from matplotlib.colors import to_hex



def plot_feature_counts_per_type(
    filtered_data, features, acronym="", top=False, ax=None, title_prefix="",
    order=None, feature_colors=None, ylim=None, hide_ylabel=False,
    label_size=25, title_size=24, textprops=18, legend_size=None,
    ):
    """
    Plot a stacked bar chart of feature presence counts across GPT-predicted project types.

    This function visualizes how often specific repository features (e.g., README, license, contributing guide)
    appear across different project types predicted by a GPT model. Each feature is represented by a stacked bar,
    segmented by project type. A special bar labeled "Project Type" can be included to show the total number of
    repositories per project type. Percentage labels are displayed above each bar, and a legend identifies
    the project type segments.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame that includes columns for the specified `features` and a 'type_prediction_gpt_5_mini' column
        representing predicted project types.
    features : list of str
        List of feature column names to include in the plot (e.g., 'readme', 'license').
    acronym : str, optional
        Acronym to include in the saved file name and title (e.g., institution name).
    top : bool, default=False
        If True, saves the plot as a PNG file in the 'plots/{acronym}' directory.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.
    title_prefix : str, optional
        Custom text to prepend to the plot title.
    order : list of str, optional
        Custom order for the features on the x-axis. If None, features are ordered by total count.
    feature_colors : dict, optional
        (Currently unused) Dictionary mapping features to specific colors.
    ylim : tuple of float, optional
        (Currently unused) Custom y-axis limits.
    hide_ylabel : bool, default=False
        Whether to hide the y-axis label and ticks.
    label_size : int, default=25
        Font size for axis labels.
    title_size : int, default=24
        Font size for the plot title.
    legend_size : int, optional
        Font size for legend text and title. If None, uses textprops.

    Returns
    -------
    tuple
        order : list of str
            The order of feature labels used in the x-axis.
        category_colors : dict
            Mapping from GPT-predicted project types to color values used in the plot.
    """

    feature_display_names = {
        'description': 'Description',
        'readme': 'README',
        'license': 'License',
        'code_of_conduct_file': 'Code of Conduct',
        'contributing': 'Contributing Guide',
        'security_policy': 'Security Policy',
        'issue_templates': 'Issue Templates',
        'pull_request_template': 'PR Template',
        'type_prediction_gpt_5_mini': "Project Type"
        
    }

    # Validate the type_prediction_gpt_5_mini column
    if "type_prediction_gpt_5_mini" not in filtered_data.columns:
        raise ValueError("filtered_data must contain a 'type_prediction_gpt_5_mini' column")

    # Map original feature names to display names
    data = filtered_data.copy()
    data = data[features + ['type_prediction_gpt_5_mini']]
    data = data.rename(columns=feature_display_names)

    # Compute grouped counts
    grouped = data.groupby('Project Type').apply(lambda g: g.notna().sum())
    grouped = grouped.T  # Features as rows, categories as columns

    if order is None:
        total_counts = grouped.sum(axis=1)
        order = total_counts.sort_values().index.tolist()

    grouped = grouped.loc[order]

    # Default colors for each type_prediction_gpt_5_mini
    category_list = grouped.columns.tolist()
    cmap = plt.colormaps['tab20']
    category_colors = {cat: cmap(i) for i, cat in enumerate(category_list)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Compute x positions with spacing before 'Project Type'
    gap = 0.6
    x_positions = []
    for i, feature in enumerate(order):
        if feature == "Project Type":
            x_positions.append(i + gap)
        else:
            x_positions.append(len(x_positions))

    # Initialize bottom values aligned with x_positions
    bottoms_array = [0] * len(order)
    used_labels = set()
    
    for category in category_list:
        for i, feature in enumerate(order):
            value = grouped.loc[feature, category]
            if value > 0:
                ax.bar(
                    x_positions[i], value,
                    bottom=bottoms_array[i],
                    label=category if category not in used_labels else "",
                    color=category_colors[category],
                    width=0.8
                )
                bottoms_array[i] += value
                used_labels.add(category)

    total_repositories = len(filtered_data)

    # Add percentage labels on top of each full bar
    for i, feature in enumerate(order):
        count = grouped.loc[feature].sum()
        percent = (count / total_repositories) * 100
        ax.annotate(f'{percent:.1f}%',
                    (x_positions[i], count + total_repositories * 0.03),
                    ha='center', va='bottom', fontsize=textprops, color='black')


    # Draw a vertical dashed line to separate 'Project Type'
    if "Project Type" in order:
        pt_index = order.index("Project Type")
        pt_x = x_positions[pt_index] - 0.8
        ax.axvline(x=pt_x, linestyle='--', color='gray', linewidth=2)

    # Formatting
    
    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{Community\ Files}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
    
    ax.set_xlabel("Community File", fontsize=label_size)
    
    if not hide_ylabel:
        ax.set_ylabel("Repository Count", fontsize=label_size)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=textprops)
    ax.tick_params(axis='y', labelsize=textprops)

    max_height = max(bottoms_array)
    ax.set_ylim(0, max_height + total_repositories * 0.13)
    ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.8)
    legend_fontsize = legend_size if legend_size is not None else textprops
    ax.legend(title="Project Type", fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    if top:
        plt.savefig(f'plots/{acronym}/CountPerFeatureTop_Stacked.png', dpi=300)

    return order, category_colors


def plot_feature_counts_per_type_altair(
    filtered_data,
    features,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=9,
):
    """Altair stacked bar chart of feature presence counts across GPT-predicted project types."""
    width = "container"
    height = "container"

    feature_display_names = {
        'description': 'Description',
        'readme': 'README',
        'license': 'License',
        'code_of_conduct_file': 'Code of Conduct',
        'contributing': 'Contributing Guide',
        'security_policy': 'Security Policy',
        'issue_templates': 'Issue Templates',
        'pull_request_template': 'PR Template',
        'type_prediction_gpt_5_mini': "Project Type",
    }

    if (
        filtered_data is None
        or (hasattr(filtered_data, "empty") and filtered_data.empty)
        or "type_prediction_gpt_5_mini" not in filtered_data.columns
    ):
        return (
            alt.Chart(pd.DataFrame({"feature": [], "project_type": [], "count": []}))
            .mark_bar()
            .properties(width=width, height=height, title="Community Files")
        )

    total_repositories = len(filtered_data)
    data = filtered_data[features + ["type_prediction_gpt_5_mini"]].copy()
    data = data.rename(columns=feature_display_names)

    grouped = data.groupby("Project Type").apply(lambda g: g.notna().sum())
    grouped = grouped.T  # features as rows, project types as columns

    total_counts = grouped.sum(axis=1)
    order = total_counts.sort_values().index.tolist()
    grouped = grouped.loc[order]

    category_list = grouped.columns.tolist()
    cmap = plt.colormaps["tab20"]
    palette = [to_hex(cmap(i)) for i in range(len(category_list))]
    color_scale = alt.Scale(domain=category_list, range=palette)

    # Build long-format dataframe
    rows = []
    for feature in order:
        for proj_type in category_list:
            rows.append({
                "feature": feature,
                "project_type": proj_type,
                "count": int(grouped.loc[feature, proj_type]),
            })
    long_df = pd.DataFrame(rows)

    # Totals per feature for percentage labels
    totals = long_df.groupby("feature")["count"].sum().reset_index()
    totals["pct"] = totals["count"].apply(
        lambda c: f"{c / total_repositories * 100:.1f}%"
    )

    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "feature:N",
                sort=order,
                title="Community File",
                axis=alt.Axis(labelAngle=-40, labelFontSize=label_size),
            ),
            y=alt.Y(
                "count:Q",
                title="Repository Count",
                stack="zero",
                axis=alt.Axis(grid=True, labelFontSize=label_size),
            ),
            color=alt.Color(
                "project_type:N",
                scale=color_scale,
                title="Project Type",
                legend=alt.Legend(labelFontSize=label_size, titleFontSize=label_size),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("project_type:N", title="Project Type"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
    )

    labels = (
        alt.Chart(totals)
        .mark_text(
            align="center", baseline="bottom", dy=-4, fontSize=textprops, color="black"
        )
        .encode(
            x=alt.X("feature:N", sort=order),
            y=alt.Y("count:Q"),
            text="pct:N",
        )
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
