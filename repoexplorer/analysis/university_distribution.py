#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_university_distribution(
    filtered_data, acronym, ax=None, color_map=None, 
    title_prefix=None, label_size=25, title_size=24, textprops=15):
    """
    Plots a pie chart representing the distribution of universities for repositories.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        The filtered dataset containing the 'university' column.

    acronym : str
        Acronym for the institution or group being plotted.

    ax : matplotlib.axes.Axes, optional
        An existing matplotlib axis to plot on. If None, creates a new figure.

    color_map : dict, optional
        A dictionary mapping university labels to colors.

    title_prefix : str, optional
        Text to prefix the plot title with.
    """

    total_repositories = len(filtered_data)

    # Count how many repos per university
    university_counts = filtered_data['university'].value_counts()

    labels = university_counts.index.tolist()

    # Color map
    cmap = cm.get_cmap('tab20')
    university_colors = {uni: cmap(i) for i, uni in enumerate(labels)}
    colors = [university_colors[uni] for uni in labels]

    # Prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        university_counts,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': textprops}
    )

    # Title
    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{UC\ University\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
