#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    # Shorten long language names
    LANGUAGE_LABEL_MAP = {
        "Jupyter Notebook": "Jupyter",
    }

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
