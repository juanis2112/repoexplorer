#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared, fixed color mapping for project types.

All plots that color-code by ``type_prediction_gpt_5_mini`` should import
``TYPE_COLOR_MAP`` (or call ``type_color_scale``) from here so that the same
type always renders in the same color regardless of the plot or the runtime
order Polars returns from a ``group_by`` / ``pivot``.
"""

import altair as alt

# Canonical project types assigned in alphabetical order against matplotlib's
# tab20 colormap (pairs of dark/light per hue), matching the original dashboard.
# tab20:  [0]blue-dk [1]blue-lt [2]orange-dk [3]orange-lt [4]green-dk [5]green-lt …
TYPE_COLOR_MAP: dict[str, str] = {
    "DATA":  "#1f77b4",  # tab20[0] blue dark
    "DEV":   "#aec7e8",  # tab20[1] blue light
    "DOCS":  "#ff7f0e",  # tab20[2] orange dark
    "EDU":   "#ffbb78",  # tab20[3] orange light
    "OTHER": "#2ca02c",  # tab20[4] green dark
    "WEB":   "#98df8a",  # tab20[5] green light
    "error": "#d62728",  # tab20[6] red dark (rarely shown)
}

# Canonical stacking order (bottom → top in stacked bars)
TYPE_ORDER: list[str] = ["DATA", "DEV", "DOCS", "EDU", "OTHER", "WEB", "error"]

# Fallback palette for any type not in the map above
_FALLBACK = [
    "#e377c2", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]


def type_sort_key(type_name: str) -> int:
    """Return the canonical stacking-order index for a project type."""
    try:
        return TYPE_ORDER.index(type_name)
    except ValueError:
        return len(TYPE_ORDER)


def type_color_scale(type_list: list[str]) -> alt.Scale:
    """Return an Altair color Scale with fixed colors for each project type.

    Types present in TYPE_COLOR_MAP get their canonical color.
    Any unexpected type falls back to a secondary palette.
    """
    colors = []
    fallback_idx = 0
    for t in type_list:
        if t in TYPE_COLOR_MAP:
            colors.append(TYPE_COLOR_MAP[t])
        else:
            colors.append(_FALLBACK[fallback_idx % len(_FALLBACK)])
            fallback_idx += 1
    return alt.Scale(domain=type_list, range=colors)
