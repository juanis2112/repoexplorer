#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text

def plot_license_distribution(    filtered_data, acronym="", ax=None, color_map=None,
    title_prefix="", hide_ylabel=False, license_order=None,
    ylim=None, label_size=25, title_size=24, textprops=16, other_thres=0.02,
    legend_size=None
):
    """
    Plots pie charts representing the distribution of licenses used in the projects:
    - Grouped license distribution (major licenses with at least 2% usage).
    - Minor license distribution (licenses with less than 2% usage).

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        The filtered dataset containing the project license data.

    Returns
    -------
    None
        This function generates and saves pie charts but does not return any values.
    """
    total_repositories = len(filtered_data)
    license_counts = filtered_data['license'].fillna('None').value_counts()
    total_licenses = license_counts.sum()
    # Group licenses
    lic_major = license_counts[license_counts / total_licenses >= 0.02].copy()
    lic_minor = license_counts[license_counts / total_licenses < 0.02].copy()
    lic_grouped = lic_major.copy()
    if not lic_minor.empty:
        lic_grouped['Other'] = lic_minor.sum()
        
    
    # Merge 'other' into 'Other' if it exists
    if 'other' in lic_grouped:
        lic_grouped['Other'] = lic_grouped.get('Other', 0) + lic_grouped['other']
        lic_grouped = lic_grouped.drop('other')
        
    labels = lic_grouped.index.tolist()
    cmap = cm.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(labels)}
    colors = [category_colors[cat] for cat in labels] # fallback: hot pink

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        lic_grouped,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': textprops},
        labeldistance=1.1,
        pctdistance=0.8
    )

    for text in texts:
        text.set_fontsize(label_size)
    for autotext in autotexts:
        autotext.set_fontsize(textprops)


    # Automatically nudge label positions to avoid overlap
    adjust_text(texts, ax=ax, ensure_inside_axes=True, expand_axes=False)
    adjust_text(autotexts, ax=ax, ensure_inside_axes=True, expand_axes=False)

    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{License\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
      
    
    # License Distribution - Grouped Plot
    # plt.figure(figsize=(8, 8))
    # plt.pie(lic_grouped, labels=lic_grouped.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8})
    # plt.title(f"{acronym.upper()} License Distribution (Grouped) Total Repositories: {total_repositories}")
    # plt.savefig(f'plots/{acronym}/license_distribution_grouped.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # # License Distribution - Minor Plot
    # if not lic_minor.empty:
    #     total_repositories = lic_minor.sum()
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     wedges, texts, autotexts = ax.pie(lic_minor, labels=lic_minor.index, autopct='%1.1f%%', startangle=140)
    #     for text in texts:
    #         text.set_fontsize(8)
    #     for i, autotext in enumerate(autotexts):
    #         autotext.set_fontsize(8)
    #         percentage = (lic_minor.iloc[i] / total_licenses) * 100
    #         autotext.set_text(f'{percentage:.1f}%')
    #     ax.set_title(f"{acronym.upper()} License Distribution (Minor Categories) Total Repositories: {total_repositories}")
    #     plt.savefig(f'plots/{acronym}/license_distribution_minor.png', dpi=300, bbox_inches='tight')
    #     plt.close()
