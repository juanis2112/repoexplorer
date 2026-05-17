#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex
import pandas as pd

LANGUAGE_LABEL_MAP = {
    "Jupyter Notebook": "Jupyter",
}

def plot_language_distribution(filtered_data, acronym="", ax=None, color_map=None,
    title_prefix="", hide_ylabel=False, language_order=None,
    ylim=None, label_size=25, title_size=24, props=12, other_thres=0.02,
    legend_size=None):
    """
    Plots pie charts representing the distribution of programming languages used in the projects:
    - Grouped language distribution (major languages with at least 2% usage).
    - Minor language distribution (languages with less than 2% usage).

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        The filtered dataset containing the project languages.

    Returns
    -------
    None
        This function generates and saves pie charts but does not return any values.
    """
    total_repositories = len(filtered_data)

    # Apply replacements
    filtered_data['language'] = filtered_data['language'].replace(LANGUAGE_LABEL_MAP)
    language_counts = filtered_data['language'].value_counts()
    total_languages = language_counts.sum()

    lang_major = language_counts[language_counts / total_languages >= 0.05].copy()
    lang_minor = language_counts[language_counts / total_languages < 0.05].copy()

    lang_grouped = lang_major.copy()
    if not lang_minor.empty:
        lang_grouped['Other'] = lang_minor.sum()

    labels = lang_grouped.index.tolist()
    cmap = cm.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(labels)}
    colors = [category_colors[cat] for cat in labels] # fallback: hot pink
 # hot pink if missing

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        lang_grouped,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': props}
    )

    for text in texts:
        text.set_fontsize(label_size)
    for autotext in autotexts:
        autotext.set_fontsize(props)

    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{Language\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )

    
    # # Language Distribution - Minor Plot
    # if not lang_minor.empty:
    #     total_repositories = lang_minor.sum()
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     wedges, texts, autotexts = ax.pie(lang_minor, labels=lang_minor.index, autopct='%1.1f%%', startangle=140)
    #     for text in texts:
    #         text.set_fontsize(8)
    #     for i, autotext in enumerate(autotexts):
    #         autotext.set_fontsize(8)
    #         percentage = (lang_minor.iloc[i] / total_languages) * 100
    #         autotext.set_text(f'{percentage:.1f}%')
    #     ax.set_title(f"{acronym.upper()} Language Distribution (Minor Categories) — Total Repositories: {total_repositories}")
    #     plt.savefig(f'plots/{acronym}/language_distribution_minor.png', dpi=300, bbox_inches='tight')
    #     plt.close()


def plot_language_distribution_altair(
    filtered_data,
    acronym="",
    label_size=10,
    title_size=12,
    textprops=8,
    other_thres=0.05,
):
    """
    Altair version of the language distribution pie chart for on-screen
    rendering. Mirrors the matplotlib version in `plot_language_distribution`.
    """
    width = "container"
    height = "container"

    if (
        filtered_data is None
        or filtered_data.empty
        or "language" not in filtered_data.columns
    ):
        return (
            alt.Chart(pd.DataFrame({"Language": [], "Count": []}))
            .mark_arc()
            .properties(width=width, height=height, title="Language Distribution")
        )

    total_repositories = len(filtered_data)
    languages = filtered_data["language"].replace(LANGUAGE_LABEL_MAP)
    language_counts = languages.value_counts()
    total_languages = language_counts.sum()

    if language_counts.empty or total_languages == 0:
        return (
            alt.Chart(pd.DataFrame({"Language": [], "Count": []}))
            .mark_arc()
            .properties(width=width, height=height, title="Language Distribution")
        )

    lang_major = language_counts[language_counts / total_languages >= other_thres].copy()
    lang_minor = language_counts[language_counts / total_languages < other_thres].copy()

    lang_grouped = lang_major.copy()
    if not lang_minor.empty:
        lang_grouped["Other"] = lang_minor.sum()

    labels = lang_grouped.index.tolist()
    cmap = cm.get_cmap("tab20")
    palette = [to_hex(cmap(i)) for i in range(len(labels))]
    color_scale = alt.Scale(domain=labels, range=palette)

    plot_df = pd.DataFrame(
        {
            "Language": labels,
            "Count": [int(lang_grouped[l]) for l in labels],
        }
    )
    plot_df["PercentLabel"] = plot_df["Count"].apply(
        lambda c: f"{(c / total_languages) * 100:.1f}%"
    )

    tooltip = [
        alt.Tooltip("Language:N", title="Language"),
        alt.Tooltip("Count:Q", title="Count"),
        alt.Tooltip("PercentLabel:N", title="Share"),
    ]

    base = alt.Chart(plot_df).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color(
            "Language:N",
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

    title = f"Language Distribution (Total: {total_repositories})"
    if acronym:
        title = f"{acronym} {title}"

    return (
        (arcs + pct_text)
        .properties(width=width, height=height, title=title)
        .configure_title(fontSize=title_size, anchor="middle")
        .configure_view(stroke=None)
    )
