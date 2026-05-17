#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import altair as alt
import random

def plot_feature_counts(
    filtered_data, features, acronym, top=False, ax=None, title_prefix=None,
    order=None, feature_colors=None, ylim=None, hide_ylabel=False,
    label_size=25, title_size=24, textprops=18, legend_size=None,
):
    """
    Plot the presence of key repository features (e.g., README, license) as a bar chart.
    
    This function counts how many repositories in the given DataFrame contain each of a 
    predefined set of features.
    
    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame containing repository metadata with boolean or non-null indicators for features.
    acronym : str
        Acronym for the university or institution (used in the plot title and file name).
    top : bool, optional
        If True, saves the figure to a file in the `plots/{acronym}/` directory. Default is False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to draw the plot on. If None, a new figure and axes are created.
    title_prefix : str, optional
        Optional text prefix for the plot title (e.g., "UC").
    order : list of str, optional
        List specifying the order in which features should appear on the x-axis. If None, features 
        are sorted by count.
    feature_colors : dict, optional
        Dictionary mapping feature display names to colors. If None, a shuffled colormap is used.
    ylim : float, optional
        Y-axis upper limit to enforce consistent scale across plots. If None, axis scales automatically.
    hide_ylabel : bool, optional
        If True, hides the y-axis label and ticks (useful for subplots). Default is False.
    label_size : int, optional
        Font size for axis labels. Default is 25.
    title_size : int, optional
        Font size for the plot title. Default is 24.
    textprops : int, optional
        Font size for percentage labels and tick labels. Default is 18.
    legend_size : int, optional
        Font size for legend (unused in this plot). Default is None.

    Returns
    -------
    order : list of str
        The order of features used in the plot.
    feature_colors : dict
        The mapping of feature display names to their corresponding colors used in the plot.
    """
    
    feature_display_names = {
        'description': 'Description',
        'readme': 'README',
        'license': 'License',
        'code_of_conduct_file': 'Code of Conduct',
        'contributing': 'Contributing Guide',
        'security_policy': 'Security Policy',
        'issue_templates': 'Issue Templates',
        'pull_request_template': 'PR Template'
    }

    # Count non-null values
    feature_counts_raw = filtered_data[features].notna().sum()
    total_repositories = len(filtered_data)

    # Map to display names
    feature_counts = pd.Series({feature_display_names[k]: v for k, v in feature_counts_raw.items()})

    # Use order from first university if provided
    if order is None:
        order = feature_counts.sort_values().index.tolist()

    # Set consistent colors across subplots
    if feature_colors is None:
        cmap = plt.get_cmap('tab10')
        palette = [cmap(i) for i in range(10)]
        random.seed("39")
        random.shuffle(palette) 
        palette = palette[0:8]
        feature_colors = dict(zip(order, palette))

    # Sort by defined order
    feature_counts = feature_counts[order]
    colors = [feature_colors[feat] for feat in order]

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    # Plot bars with matplotlib
    bars = ax.bar(feature_counts.index, feature_counts.values, color=colors)

    # Add percentage labels on top of bars
    for bar in bars:
        count = bar.get_height()
        percent = (count / total_repositories) * 100
        ax.annotate(f'{percent:.1f}%',
                    (bar.get_x() + bar.get_width() / 2, count + total_repositories * 0.015),
                    ha='center', va='bottom', fontsize=textprops, color='black')

    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{Community\ Files\ Presence}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
    ax.set_xlabel("Feature", fontsize=label_size)
    if not hide_ylabel:
        ax.set_ylabel("Repository Count", fontsize=label_size)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)

    ax.set_xticks(range(len(feature_counts.index)))
    ax.set_xticklabels(feature_counts.index, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)

    # Set consistent ylim across plots, with a bit of headroom for labels
    if ylim is not None:
        ax.set_ylim(0, ylim * 1.05)
    else:
        bottom, topy = ax.get_ylim()
        ax.set_ylim(bottom, topy * 1.05)
    ax.set_xlim(-0.5, len(features)-0.3)

    # Save
    if top:
        plt.savefig(f'plots/{acronym}/CountPerFeatureTop.png', dpi=300)

    return order, feature_colors

def plot_feature_counts_altair(
    data,
    features,
    acronym="",
    label_size=8,
    title_size=12,
    textprops=8,
):
    """
    Altair version of the combined feature-counts chart for on-screen rendering
    and PNG download.
    """
    feature_display_names = {
        "description": "Description",
        "readme": "README",
        "license": "License",
        "code_of_conduct_file": "Code of Conduct",
        "contributing": "Contributing Guide",
        "security_policy": "Security Policy",
        "issue_templates": "Issue Templates",
        "pull_request_template": "PR Template",
    }

    width = "container"
    height = "container"

    if data is None or data.empty:
        return (
            alt.Chart(pd.DataFrame({"Feature": [], "Count": []}))
            .mark_bar()
            .properties(width=width, height=height, title="Community Files Presence")
        )

    feature_counts_raw = data[features].notna().sum()
    total_repositories = len(data)
    feature_counts = pd.Series(
        {feature_display_names[k]: v for k, v in feature_counts_raw.items()}
    )
    order = feature_counts.sort_values().index.tolist()

    cmap = plt.get_cmap("tab10")
    palette = [to_hex(cmap(i)) for i in range(10)]
    random.seed("39")
    random.shuffle(palette)
    palette = palette[: len(order)]
    color_scale = alt.Scale(domain=order, range=palette)

    plot_df = pd.DataFrame(
        {"Feature": order, "Count": [int(feature_counts[f]) for f in order]}
    )
    plot_df["PercentLabel"] = plot_df["Count"].apply(
        lambda c: f"{(c / total_repositories) * 100:.1f}%"
    )
    y_max = plot_df["Count"].max() * 1.12
    plot_df["FullHeight"] = y_max

    tooltip = [
        alt.Tooltip("Feature:N", title="Feature"),
        alt.Tooltip("Count:Q", title="Count"),
        alt.Tooltip("PercentLabel:N", title="Share"),
    ]

    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Feature:N",
                sort=order,
                title="Feature",
                axis=alt.Axis(labelAngle=-45, labelFontSize=label_size),
            ),
            y=alt.Y(
                "Count:Q",
                title="Repository Count",
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(grid=True, labelFontSize=label_size),
            ),
            color=alt.Color("Feature:N", scale=color_scale, legend=None),
        )
    )

    # Transparent full-height hit areas so tooltips snap to the column
    # even when the actual bar is tiny.
    hit_area = (
        alt.Chart(plot_df)
        .mark_bar(opacity=0)
        .encode(
            x=alt.X("Feature:N", sort=order),
            y=alt.Y("FullHeight:Q", scale=alt.Scale(domain=[0, y_max])),
            tooltip=tooltip,
        )
    )

    labels = (
        alt.Chart(plot_df)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-4,
            color="black",
            fontSize=textprops,
        )
        .encode(
            x=alt.X("Feature:N", sort=order),
            y=alt.Y("Count:Q"),
            text="PercentLabel:N",
        )
    )

    title = f"Community Files Presence (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (bars + labels + hit_area)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_axis(titleFontSize=label_size)
        .configure_view(stroke=None)
    )
