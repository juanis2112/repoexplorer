#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import polars as pl
from shiny.express import input, ui, render
from shiny import reactive
from shiny import session as shiny_session
from shiny import ui as sui
from shinywidgets import render_altair
from faicons import icon_svg
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import logging
import querychat as qc
from repoexplorer.analysis.type_distribution import plot_type_distribution, plot_type_distribution_altair
from repoexplorer.analysis.language_distribution_by_type import plot_language_distribution_by_type, plot_language_distribution_by_type_altair
from repoexplorer.analysis.language_distribution import plot_language_distribution, plot_language_distribution_altair
from repoexplorer.analysis.license_distribution_by_type import plot_license_distribution_by_type, plot_license_distribution_by_type_altair
from repoexplorer.analysis.license_distribution import plot_license_distribution, plot_license_distribution_altair
from repoexplorer.analysis.feature_counts_per_type import plot_feature_counts_per_type, plot_feature_counts_per_type_altair
from repoexplorer.analysis.feature_counts import plot_feature_counts, plot_feature_counts_altair
from repoexplorer.analysis.university_distribution import plot_university_distribution
from repoexplorer.analysis.feature_heatmap_per_stars import plot_feature_heatmap_by_star_bucket, plot_feature_heatmap_by_star_bucket_altair
from repoexplorer.analysis.commit_history import plot_commit_history
from repoexplorer.analysis.stars_distribution_bar import plot_stars_distribution_bar_altair
from repoexplorer.analysis.forks_distribution_bar import plot_forks_distribution_bar_altair
from repoexplorer.analysis.release_downloads_distribution_bar import plot_release_downloads_distribution_bar_altair
from repoexplorer.analysis.contributors_distribution_bar import plot_contributors_distribution_bar_altair
from repoexplorer.analysis.bus_factor_distribution_bar import plot_bus_factor_distribution_bar_altair
from repoexplorer.analysis.contributor_count_bucket_bar import plot_contributor_count_bucket_bar_altair
from dotenv import load_dotenv

load_dotenv()


# Global feature flag: enable/disable the chat tab and all chat behavior.
# By default this is False; set the environment variable ENABLE_CHAT=true
# (or edit this value) to turn the chat tab back on.
ENABLE_CHAT = False


# Global flag for where to read data:
# - "local"  -> use parquet files under Data/parquet (default)
# - "remote" -> download reduced combined parquet files from S3 bucket
DATA = os.getenv("DATA")


if "OPENAI_MODEL" not in os.environ:
    os.environ["OPENAI_MODEL"] = "gpt-5-mini"

# Data/parquet/{acronym}/repositories.parquet (case-insensitive acronym match)
# Optional fast path: Data/parquet/repositories_combined.parquet (single pre-merged file)
PARQUET_BASE = "Data/parquet"
# COMBINED_PARQUET = os.path.join(PARQUET_BASE, "repositories_combined_clean.parquet")
# SECURITY_PARQUET = os.path.join(PARQUET_BASE, "security_combined_clean.parquet")

COMBINED_PARQUET = os.path.join(PARQUET_BASE, "repositories_reduced_combined_stars_gt_0.parquet")
SECURITY_PARQUET = os.path.join(PARQUET_BASE, "security_reduced_combined_stars_gt_0.parquet")
ORGANIZATIONS_PARQUET = "../repofinder/Data/parquet/organizations.parquet"
# CONTRIBUTORS_PARQUET = os.path.join(PARQUET_BASE, "contributors_combined_clean.parquet")
# COMMITS_PARQUET = os.path.join(PARQUET_BASE, "commits_combined_clean.parquet")

# Columns to load (fewer columns = faster load). "university" is added from config.
COLUMNS_TO_LOAD = [
    "university", "id", "full_name", "owner", "license", "language", "html_url", "description", "fork", "created_at",
    "updated_at", "pushed_at", "homepage", "size", "stargazers_count", "readme",
    "watchers_count", "forks_count", "open_issues_count", "watchers", "organization", "release_downloads", "contributors", 
    "contributor_count", "bus_factor", "code_of_conduct_file", "contributing", "security_policy", "issue_templates",
    "pull_request_template", "subscribers_count", "affiliation_prediction_gpt_5_mini", "type_prediction_gpt_5_mini",
]

ACRONYMS = [
    "UCB", "UCI", "UCD", "UCLA", "UCM", "UCR", "UCSB", "UCSC", "UCSD", "UCSF",
    "Biohub", "CMU", "ETH", "GWU", "Lero", "MGB", "MSU", "OSU", "RIT", "SLU",
    "Syracuse", "TCD", "UGA", "SnT", "UCL", "UMich", "UVM", "UWMadison", "JHU",
    "Georgia Tech", "UT Austin", "Stanford",
]

FEATURES = [
    'description', 'readme', 'license', 'code_of_conduct_file',
    'contributing', 'security_policy', 'issue_templates', 'pull_request_template'
]

# OpenSSF scorecard-style columns in ``df_security`` (joined to repos on ``html_url``).
# Display label, parquet column name (same as repository detail Security tab).
SECURITY_SCORECARD_METRICS = [
    ("Binary artifacts", "Binary_Artifacts"),
    ("Branch protection", "Branch_Protection"),
    ("CI tests", "CI_Tests"),
    ("CII Best Practices", "CII_Best_Practices"),
    ("Code review", "Code_Review"),
    ("Contributors", "Contributors"),
    ("Dangerous workflow", "Dangerous_Workflow"),
    ("Dependency update tool", "Dependency_Update_Tool"),
    ("Fuzzing", "Fuzzing"),
    ("License (scorecard)", "License"),
    ("Maintained", "Maintained"),
    ("Packaging", "Packaging"),
    ("Pinned dependencies", "Pinned_Dependencies"),
    ("SAST", "SAST"),
    ("Security policy (scorecard)", "Security_Policy"),
    ("Signed releases", "Signed_Releases"),
    ("Token permissions", "Token_Permissions"),
    ("Vulnerabilities", "Vulnerabilities"),
    ("Total score", "Total_Score"),
]


def _is_missing_scalar(v):
    """True for None or NaN (common null representations from Polars rows)."""
    if v is None:
        return True
    try:
        return v != v  # NaN != NaN is True for float('nan')
    except Exception:
        return False


def _safe_markdown_text(v):
    """ui.markdown / textwrap require str; float NaN from pandas breaks deploy."""
    if _is_missing_scalar(v):
        return ""
    return str(v)


def _safe_display_str(v, default="—"):
    if _is_missing_scalar(v):
        return default
    return str(v)


def _safe_int_metric(v):
    if _is_missing_scalar(v):
        return "N/A"
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return "N/A"


def _format_thousands_approx(n) -> str:
    """Round counts for display (e.g. 52_000 -> '52K', 900 -> '900')."""
    if _is_missing_scalar(n):
        return "—"
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "—"
    if x != x:  # NaN check
        return "—"
    x = int(round(x))
    if x == 0:
        return "0"
    sign = ""
    if x < 0:
        sign = "-"
        x = abs(x)
    if x < 1000:
        return f"{sign}{x}"
    k = int(round(x / 1000.0))
    return f"{sign}{k}K"


def _has_nonempty_text(v):
    if _is_missing_scalar(v):
        return False
    s = str(v).strip()
    return len(s) > 0 and s.lower() not in ("none", "nan", "<na>")


def _truthy_feature_flag(v):
    """For 0/1 or boolean presence columns; NaN is false."""
    if _is_missing_scalar(v):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return float(v) != 0.0
        except Exception:
            return False
    s = str(v).strip().lower()
    return s not in ("", "none", "nan", "false", "0", "<na>")


def _normalize_license_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a new DataFrame with missing / empty licenses replaced with null.
    """
    if df is None or df.is_empty() or "license" not in df.columns:
        return df
    return df.with_columns(
        pl.when(
            pl.col("license").is_null()
            | pl.col("license").str.strip_chars().str.to_lowercase().is_in(
                ["", "none", "nan", "null", "<na>"]
            )
        )
        .then(None)
        .otherwise(pl.col("license").str.strip_chars())
        .alias("license")
    )


def _make_feature_counts_combined_fig(
    data,
    features,
    acronym="",
    figsize=(8, 6),
    label_size=8,
    title_size=12,
    textprops=8,
):
    """
    Helper to build the combined feature-counts figure so it can be reused
    both for on-screen rendering and download without duplicating code.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_feature_counts(
        data,
        features,
        acronym=acronym,
        ax=ax,
        label_size=label_size,
        title_size=title_size,
        textprops=textprops,
    )
    return fig


def _make_license_combined_fig(
    data,
    acronym: str = "",
    figsize=(8, 6),
    label_size: int = 10,
    title_size: int = 12,
    textprops: int = 7,
    other_thres: float = 0.009,
):
    """
    Helper to build the combined license distribution figure.
    Reused for on-screen rendering and download.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_license_distribution(
        data,
        acronym=acronym,
        ax=ax,
        label_size=label_size,
        title_size=title_size,
        textprops=textprops,
        other_thres=other_thres,
    )
    return fig


def _make_language_combined_fig(
    data,
    acronym: str = "",
    figsize=(8, 6),
    label_size: int = 10,
    title_size: int = 12,
    props: int = 9,
    other_thres: float = 0.1,
):
    """
    Helper to build the combined language distribution figure.
    Reused for on-screen rendering and download.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_language_distribution(
        data,
        acronym=acronym,
        ax=ax,
        label_size=label_size,
        title_size=title_size,
        props=props,
        other_thres=other_thres,
    )
    return fig


# Read data from public bucket
def read_parquet_from_s3_public(bucket_name, object_key, columns=None):
    url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    return pl.read_parquet(url, columns=columns)

def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    int_cols = [
        "stargazers_count", "forks_count", "watchers_count",
        "open_issues_count", "subscribers_count", "contributor_count",
        "bus_factor", "release_downloads",
    ]
    exprs = [
        pl.col(c).cast(pl.Int32, strict=False)
        for c in int_cols if c in df.columns
    ]
    if "affiliation_prediction_gpt_5_mini" in df.columns:
        exprs.append(pl.col("affiliation_prediction_gpt_5_mini").cast(pl.Float32, strict=False))
    if exprs:
        df = df.with_columns(exprs)
    return df

#------------------------------------ Styling ---------------------------------------------
# Add CSS for hover tooltip
ui.tags.style("""
.repo-data-card {
    position: relative;
}
.repo-data-card::after {
    content: "Click a row to see repository details";
    position: absolute;
    top: 10px;
    right: 10px;
    background-color: rgba(0, 0, 0, 0.85);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 1000;
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.repo-data-card:hover::after {
    opacity: 1;
}
""")

ui.tags.style("""
.nav-pills .nav-link,
.nav-tabs .nav-link {
    font-size: 0.9rem;   /* adjust smaller/larger as you like */
}
""")

ui.tags.style("""
/* Sidebar filter labels: Prediction Threshold, University, Project Type, License, Language, etc. */
.bslib-sidebar .form-label,
.bslib-sidebar .control-label,
aside .control-label,
aside .form-label,
[data-bslib-sidebar] .control-label,
[data-bslib-sidebar] .form-label {
    font-size: 0.85rem !important;
}
/* By id for sidebar labels (e.g. type-label, university label, etc.) */
#slider_threshold-label,
#university-label,
#type-label,
#license-label,
#language-label,
#slider_stars-label,
#slider_forks-label,
#slider_downloads-label {
    font-size: 0.85rem !important;
}
""")

ui.tags.style("""
/* DataGrid column headers: allow \n in column names to wrap to multiple lines */
.shiny-data-grid thead th {
    white-space: pre-line !important;
    text-align: center !important;
}
""")

ui.tags.style("""
/* Altair / Vega chart hover tooltips */
#vg-tooltip-element,
#vg-tooltip-element table,
#vg-tooltip-element td,
#vg-tooltip-element th {
    font-size: 15px !important;
    line-height: 1.45 !important;
}
#vg-tooltip-element {
    padding: 12px 16px !important;
}
""")

ui.tags.style("""
/* Sidebar: allow manual resize by dragging the right edge */
.bslib-sidebar,
aside[data-bslib-sidebar],
[data-bslib-sidebar] {
    resize: horizontal;
    overflow: auto;
    min-width: 200px;
    max-width: 60%;
}
""")

# Remove stray "True/False" that can appear from querychat/Shiny return values (This is hacky but leaving it for now)
ui.tags.script("""
(function() {
  function isStrayBooleanText(text) {
    var t = (text || "").trim();
    return t === "True" || t === "False";
  }
  function removeBooleanNodes(node) {
    if (!node) return;
    if (node.nodeType === Node.TEXT_NODE && isStrayBooleanText(node.textContent)) {
      node.parentNode.removeChild(node);
      return;
    }
    if (node.nodeType === Node.ELEMENT_NODE && node.childNodes.length === 1 &&
        node.childNodes[0].nodeType === Node.TEXT_NODE &&
        isStrayBooleanText(node.childNodes[0].textContent)) {
      node.parentNode.removeChild(node);
      return;
    }
    for (var i = node.childNodes.length - 1; i >= 0; i--) {
      removeBooleanNodes(node.childNodes[i]);
    }
  }
  function run() {
    removeBooleanNodes(document.body);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
  setTimeout(run, 500);
})();

// (No custom \"Viewing rows X through Y\" logic needed for universities now that we use DataGrid)
""")

#------------------------------------ Load the data ---------------------------------------------
# There is two ways to load the data:
# 1. Fast path: single pre-merged parquet 
# 2. Slow path: load each parquet file individually and merge them together


# def _load_one_acronym(acronym: str, parquet_dir: str):
#     """
#     Load a repository parquet file and add the university name.

#     Parameters
#     ----------
#     acronym : str
#         University acronym (for example, ``"UCB"``).
#     parquet_dir : str
#         Subdirectory under ``PARQUET_BASE`` that contains ``repositories.parquet``.

#     Returns
#     -------
#     pandas.DataFrame or None
#         DataFrame with the requested columns and a ``"university"`` column,
#         or ``None`` if the parquet file is missing or cannot be read.
#     """
#     repo_path = os.path.join(PARQUET_BASE, parquet_dir, "repositories.parquet")
#     if not os.path.isfile(repo_path):
#         return None
#     try:
#         # Read only needed columns for faster I/O
#         df = pd.read_parquet(repo_path, columns=[c for c in COLUMNS_TO_LOAD if c != "university"])
#     except Exception:
#         df = pd.read_parquet(repo_path)
#         df = df[[c for c in COLUMNS_TO_LOAD if c in df.columns and c != "university"]]
#     config_file = f"config/config_{acronym.replace(' ', '_')}.json"
#     if os.path.isfile(config_file):
#         try:
#             with open(config_file, encoding="utf-8") as f:
#                 df["university"] = json.load(f).get("UNIVERSITY_NAME", acronym)
#         except Exception:
#             df["university"] = acronym
#     else:
#         df["university"] = acronym
#     return df


if DATA == "remote":
    # Usage Shiny app
    _df_pl = read_parquet_from_s3_public("repoexplorer-data", "repositories_reduced_combined_stars_gt_0.parquet", columns=COLUMNS_TO_LOAD)
    _df_security_pl = read_parquet_from_s3_public("repoexplorer-data", "security_reduced_combined_stars_gt_0.parquet")
    _df_organizations_pl = read_parquet_from_s3_public("repoexplorer-data", "organizations.parquet")

else:
    # Load main repositories table
    _df_pl = pl.read_parquet(COMBINED_PARQUET, columns=COLUMNS_TO_LOAD)
    if "university" not in _df_pl.columns:
        _df_pl = _df_pl.with_columns(pl.lit("Unknown").alias("university"))
    _df_security_pl = pl.DataFrame()
    if os.path.isfile(SECURITY_PARQUET):
        try:
            _df_security_pl = pl.read_parquet(SECURITY_PARQUET)
        except Exception:
            logging.exception("Failed to load security parquet %s", SECURITY_PARQUET)
            _df_security_pl = pl.DataFrame()
    _df_organizations_pl = pl.DataFrame()
    if os.path.isfile(ORGANIZATIONS_PARQUET):
        try:
            _df_organizations_pl = pl.read_parquet(ORGANIZATIONS_PARQUET)
        except Exception:
            logging.exception("Failed to load organizations parquet %s", ORGANIZATIONS_PARQUET)
            _df_organizations_pl = pl.DataFrame()

_df_pl = optimize_dtypes(_df_pl)
df = _df_pl
df_security = _df_security_pl
df_organizations = _df_organizations_pl
del _df_pl, _df_security_pl, _df_organizations_pl

# # Load contributors table
# df_contributors = pd.DataFrame()
# if os.path.isfile(CONTRIBUTORS_PARQUET):
#     try:
#         df_contributors = pd.read_parquet(CONTRIBUTORS_PARQUET)
#     except Exception:
#         logging.exception("Failed to load contributors parquet %s", CONTRIBUTORS_PARQUET)
#         df_contributors = pd.DataFrame()

# Load commits (all_combined) table
# df_commits = pd.DataFrame()
# if os.path.isfile(COMMITS_PARQUET):
#     try:
#         df_commits = pd.read_parquet(COMMITS_PARQUET)
#     except Exception:
#         logging.exception("Failed to load commits parquet %s", COMMITS_PARQUET)
#         df_commits = pd.DataFrame()


df = _normalize_license_column(df)


# =============================================== App UI ==========================================

ui.page_opts(title="Open Source Repository Browser", fillable=True)

# ======================================== Filter options =========================================

licenses = df["license"].drop_nulls().unique().to_list() if "license" in df.columns else []
languages = df["language"].drop_nulls().unique().to_list() if "language" in df.columns else []
universities = df["university"].drop_nulls().unique().to_list() if "university" in df.columns else []
types = df["type_prediction_gpt_5_mini"].drop_nulls().unique().to_list() if "type_prediction_gpt_5_mini" in df.columns else []

# Organizations filter options
_org_unis = (
    sorted(df_organizations["university"].drop_nulls().unique().to_list())
    if not df_organizations.is_empty() and "university" in df_organizations.columns
    else []
)

# Subset with default prediction threshold (>= 0.8) for sliders and for chat
_df_08 = df.filter(pl.col("affiliation_prediction_gpt_5_mini") >= 0.8)
_m = _df_08["stargazers_count"].max() if not _df_08.is_empty() else None
_slider_max_stars = int(_m) if _m is not None else 5000
_m = _df_08["forks_count"].max() if not _df_08.is_empty() else None
_slider_max_forks = int(_m) if _m is not None else 100
_m = _df_08["release_downloads"].max() if not _df_08.is_empty() else None
_slider_max_downloads = int(_m) if _m is not None else 1000

# ------------------------------------ QueryChat Config -------------------------------------------
if ENABLE_CHAT:
    _greeting_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greeting.md")
    with open(_greeting_path, encoding="utf-8") as _f:
        _greeting_md = _f.read()

    querychat_config = qc.init(
        data_source=_df_08,
        table_name="Repositories",
        greeting=_greeting_md,
    )
else:
    querychat_config = None

# Assign chat server in a function so its return value is not rendered as "True" in the main panel
chat = None

if ENABLE_CHAT:
    def _init_chat_server():
        global chat
        chat = qc.server("chat", querychat_config)

    _init_chat_server()

# Absorb any stray top-level return value so "True" does not render in main panel
ui.HTML("")

# Reset all filters when button is clicked
@reactive.effect
@reactive.event(input.reset_filters)
def reset_all_filters():
    ui.update_selectize("university", selected=[])
    ui.update_selectize("type", selected=[])
    ui.update_selectize("license", selected=[])
    ui.update_selectize("language", selected=[])
    ui.update_slider("slider_stars", value=[0, _slider_max_stars])
    ui.update_slider("slider_forks", value=[0, _slider_max_forks])
    ui.update_slider("slider_downloads", value=[0, _slider_max_downloads])
    ui.update_text("table_search", value="")

    if ENABLE_CHAT:
        try:
            sess = shiny_session.get_current_session()
            if sess is not None:
                sess.send_input_message("chat-message", {"value": "reset all filters"})
        except Exception:
            pass

# ======================================== Constants & shared resources ============================

_OVERVIEW_TITLE_SIZE = 14
_OVERVIEW_LABEL_SIZE = 11
_OVERVIEW_TEXT_SIZE = 10
_OVERVIEW_BAR_PCT_SIZE = 11
_OVERVIEW_PIE_PCT_SIZE = 11
_TABLE_FONT_SIZE = "14px"

ICONS = {
    "repos": icon_svg("code-branch"),
    "contributors": icon_svg("users"),
    "active": icon_svg("clock"),
    "openssf": icon_svg("shield-halved"),
    "busfactor": icon_svg("bus"),
    "license": icon_svg("id-card"),
    "stars": icon_svg("star"),
    "forks": icon_svg("code-fork"),
    "downloads": icon_svg("download"),
    "organizations": icon_svg("building"),
    "link": icon_svg("link"),
    "description": icon_svg("pen-to-square"),
    "email": icon_svg("envelope"),
}

_about_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "about.md")
with open(_about_path, encoding="utf-8") as _f:
    _about_md = _f.read()

# ======================================== Main UI ================================================

with ui.navset_pill(id="main_tab", selected="Repositories"):
    # ============================= ABOUT TAB =====================================================
    with ui.nav_panel("About"):
        with ui.card():
            ui.markdown(_about_md)

    # ============================= REPOSITORIES TAB ===============================================
    with ui.nav_panel("Repositories"):
        with ui.layout_sidebar(fillable=True):
            with ui.sidebar(open="open", bg="#f8f8f8", width="250px"):
                with ui.navset_pill(id="side_tab"):
                    with ui.nav_panel("Manual Filters"):
                        ui.input_slider("slider_threshold", "Prediction Threshold", min=0, max=1, value=[0.8, 1])
                        ui.input_selectize("university", "University:", universities, multiple=True)
                        ui.input_selectize("type", "Project Type:", types, multiple=True)
                        ui.input_selectize("license", "License:", licenses, multiple=True)
                        ui.input_selectize("language", "Language:", languages, multiple=True)
                        ui.input_slider("slider_stars", "# Stars", min=0, max=_slider_max_stars, value=[0, _slider_max_stars])
                        ui.input_slider("slider_forks", "# Forks", min=0, max=_slider_max_forks, value=[0, _slider_max_forks])
                        ui.input_slider("slider_downloads", "# Release Downloads", min=0, max=_slider_max_downloads, value=[0, _slider_max_downloads])

                    if ENABLE_CHAT:
                        with ui.nav_panel("Chat Bot"):
                            qc.ui("chat")

                ui.br()
                ui.br()
                ui.input_action_button("reset_filters", "Reset All Filters", class_="btn-danger")
                ui.HTML("")

            # ---- Inner tabs ----
            with ui.navset_tab(id="repo_tab", selected="Overview"):

                # ---- Overview ----
                with ui.nav_panel("Overview"):
                        # University table + value boxes (2x2 grid)

                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                ui.markdown("**Repositories per University**")
                                @render.data_frame
                                def university_table():
                                    data = filtered_df()
                                    if "university" not in data.columns or data.is_empty():
                                        return render.DataGrid(pl.DataFrame({"University": [], "Count": []}))

                                    university_counts = (
                                        data.group_by("university")
                                        .agg(pl.len().alias("Count"))
                                        .sort("Count", descending=True)
                                        .rename({"university": "University"})
                                    )
                                    return render.DataGrid(
                                        university_counts,
                                        width="100%",
                                        height="400px",
                                        styles=[
                                            {
                                                "location": "body",
                                                "cols": [0],
                                                "style": {"minWidth": "70%", "width": "70%"},
                                            },
                                            {
                                                "location": "body",
                                                "cols": [1],
                                                "style": {"minWidth": "30%", "width": "30%", "textAlign": "right"},
                                            },
                                        ],
                                    )

                            with ui.div():
                                with ui.layout_columns(col_widths=(6, 6)):
                                    with ui.value_box(showcase=ICONS["repos"]):
                                        "Total repositories"
                                        @render.express
                                        def total_repos():
                                            len(filtered_df())

                                    with ui.value_box(showcase=ICONS["contributors"]):
                                        "Total contributors"
                                        @render.express
                                        def total_contributors():
                                            data = filtered_df()
                                            if "contributor_count" not in data.columns:
                                                "—"
                                            else:
                                                s = data["contributor_count"].cast(pl.Float64, strict=False).drop_nulls()
                                                int(s.sum()) if len(s) > 0 else 0

                                with ui.layout_columns(col_widths=(6, 6)):
                                    with ui.value_box(showcase=ICONS["license"]):
                                        "Repositories with a license"
                                        @render.express
                                        def pct_with_license():
                                            data = filtered_df()
                                            if "license" not in data.columns:
                                                "—"
                                            else:
                                                total = data.height
                                                if total == 0:
                                                    "0%"
                                                else:
                                                    with_license = data["license"].is_not_null().sum()
                                                    pct = 100.0 * with_license / total
                                                    f"{pct:.1f}%"

                                    with ui.value_box(showcase=ICONS["busfactor"]):
                                        "Average bus factor"
                                        @render.express
                                        def avg_busfactor():
                                            data = filtered_df()
                                            col = "bus_factor"
                                            if col not in data.columns:
                                                "—"
                                            else:
                                                v = data[col].cast(pl.Float64, strict=False).mean()
                                                f"{v:.1f}" if v is not None else "—"

                        # Type distribution + Community files presence
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_type():
                                    return plot_type_distribution_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_PIE_PCT_SIZE,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_files_combined():
                                    return plot_feature_counts_altair(
                                        filtered_df(),
                                        FEATURES,
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                        # Language + License distributions
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_language_combined():
                                    return plot_language_distribution_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_PIE_PCT_SIZE,
                                        other_thres=0.05,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_license_combined():
                                    return plot_license_distribution_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_PIE_PCT_SIZE,
                                        other_thres=0.02,
                                    )
        
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_license():
                                    return plot_license_distribution_by_type_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                        other_thres=0.009,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_language():
                                    return plot_language_distribution_by_type_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                        other_thres=0.02,
                                    )
    

                                # ---- Browse (repository table + detail) ----
                with ui.nav_panel("Browse"):
                        with ui.card(class_="repo-data-card"):
                            ui.tags.div(
                                ui.tags.div(
                                    ui.input_text(
                                        "table_search",
                                        "Search",
                                        placeholder="Search repositories...",
                                        width="100%",
                                    ),
                                    style="flex: 1; min-width: 220px;",
                                ), 
                            )
                            @render.data_frame
                            def display_df():
                                data = repositories_table_df()
                                return render.DataGrid(
                                    data,
                                    height="500px",
                                    selection_mode="row",
                                )

                            @render.download(
                                filename=lambda: f"repositories_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            )
                            def download_repositories_csv():
                                out_df = repositories_table_df()
                                buf = io.BytesIO()
                                out_df.write_csv(buf)
                                buf.seek(0)
                                yield buf.getvalue()
                        with ui.card():
                            @render.ui
                            def show_clicked():
                                selected_rows = display_df.cell_selection()["rows"]

                                if not selected_rows:
                                    return ""

                                # Row position in the grid matches repositories_table_df (search + column drops).
                                view = repositories_table_df()
                                row_pos = selected_rows[0]
                                # Use html_url as key to look up the full row in filtered_df()
                                _view_row = view.row(row_pos, named=True)
                                _row_url = _view_row.get("html_url")
                                _full_matches = filtered_df().filter(pl.col("html_url") == _row_url) if _row_url else pl.DataFrame()
                                selected = _full_matches.row(0, named=True) if not _full_matches.is_empty() else _view_row

                                _readme_md = _safe_markdown_text(selected.get("readme"))
                                _contributing_md = _safe_markdown_text(selected.get("contributing"))
                                _security_policy_md = _safe_markdown_text(selected.get("security_policy"))

                                # Match security metrics row (from security_combined_clean.parquet) by html_url
                                sec_row = None
                                if not df_security.is_empty() and "html_url" in df_security.columns:
                                    _matches = df_security.filter(pl.col("html_url") == selected.get("html_url"))
                                    if not _matches.is_empty():
                                        sec_row = _matches.row(0, named=True)

                                # Two-column layout:
                                # - Left column: Overview / Impact / Health / Security
                                # - Right column: README / Contributing / Security Policy
                                return sui.layout_columns(
                                    ui.div(
                                        sui.navset_tab(
                                            sui.nav_panel(
                                                "Overview",
                                                ui.p(
                                                    ui.tags.span("Name: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    _safe_display_str(selected.get("full_name")),
                                                ),
                                                ui.p(
                                                    ui.tags.span("University: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    _safe_display_str(selected.get("university"), "Unknown"),
                                                ),
                                                ui.p(
                                                    ui.tags.span("License: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    _safe_display_str(selected.get("license")),
                                                ),
                                                ui.p(
                                                    ui.tags.span("Language: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    _safe_display_str(selected.get("language")),
                                                ),
                                                ui.p(
                                                    ui.tags.span("Project Type: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    _safe_display_str(selected.get("type_prediction_gpt_5_mini")),
                                                ),
                                                ui.p(
                                                    ui.tags.span("Description: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    _safe_display_str(selected.get("description")),
                                                ),
                                                ui.p(
                                                    ui.tags.span("URL: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                                    (
                                                        ui.tags.a(
                                                            _safe_display_str(selected.get("html_url"), ""),
                                                            href=_safe_display_str(selected.get("html_url"), ""),
                                                            target="_blank",
                                                        )
                                                        if _has_nonempty_text(selected.get("html_url"))
                                                        else "—"
                                                    ),
                                                ),
                                            ),
                                            sui.nav_panel(
                                                "Impact",
                                                ui.tags.table(
                                                    ui.tags.tr(
                                                        ui.tags.th(
                                                            "Metric",
                                                            style=(
                                                                "padding-right: 6px; text-align: left; "
                                                                "font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.th("Value", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Number of stars",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td(_safe_int_metric(selected.get("stargazers_count")), style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Number of downloads",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td(_safe_int_metric(selected.get("release_downloads")), style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Number of forks",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td(_safe_int_metric(selected.get("forks_count")), style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Number of issues",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td(_safe_int_metric(selected.get("open_issues_count")), style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Number of contributors",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td(_safe_int_metric(selected.get("contributor_count")), style="text-align: center;"),
                                                    ),
                                                    style="width: 100%; border-collapse: collapse;",
                                                ),
                                            ),
                                            sui.nav_panel(
                                                "Health",
                                                ui.tags.table(
                                                    ui.tags.tr(
                                                        ui.tags.th(
                                                            "Health check",
                                                            style=(
                                                                "padding-right: 6px; text-align: left; "
                                                                "font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.th("Present", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Description",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _has_nonempty_text(selected.get("description")) else "✗", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "README",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _has_nonempty_text(selected.get("readme")) else "✗", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Contributing guide",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _has_nonempty_text(selected.get("contributing")) else "✗", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Code of conduct",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _truthy_feature_flag(selected.get("code_of_conduct_file")) else "✗", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Security policy",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _has_nonempty_text(selected.get("security_policy")) else "✗", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "Issue templates",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _truthy_feature_flag(selected.get("issue_templates")) else "✗", style="text-align: center;"),
                                                    ),
                                                    ui.tags.tr(
                                                        ui.tags.td(
                                                            "PR template",
                                                            style=(
                                                                "padding-right: 6px; "
                                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                            ),
                                                        ),
                                                        ui.tags.td("✅" if _truthy_feature_flag(selected.get("pull_request_template")) else "✗", style="text-align: center;"),
                                                    ),
                                                    style="width: 100%; border-collapse: collapse;",
                                                ),
                                            ),
                                            sui.nav_panel(
                                                "Security",
                                                (
                                                    ui.p("No security metrics available", class_="text-muted")
                                                    if sec_row is None
                                                    else ui.tags.table(
                                                        ui.tags.tr(
                                                            ui.tags.th(
                                                                "Metric",
                                                                style=(
                                                                    "padding-right: 6px; text-align: left; "
                                                                    "font-weight: bold;"
                                                                ),
                                                            ),
                                                            ui.tags.th("Value", style="text-align: center;"),
                                                        ),
                                                        *[
                                                            ui.tags.tr(
                                                                ui.tags.td(
                                                                    name,
                                                                    style=(
                                                                        "padding-right: 6px; "
                                                                        "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                                                    ),
                                                                ),
                                                                ui.tags.td(_safe_display_str(sec_row.get(col)), style="text-align: center;"),
                                                            )
                                                            for name, col in [
                                                                ("Binary artifacts", "Binary_Artifacts"),
                                                                ("Branch protection", "Branch_Protection"),
                                                                ("CI tests", "CI_Tests"),
                                                                ("CII Best Practices", "CII_Best_Practices"),
                                                                ("Code review", "Code_Review"),
                                                                ("Contributors", "Contributors"),
                                                                ("Dangerous workflow", "Dangerous_Workflow"),
                                                                ("Dependency update tool", "Dependency_Update_Tool"),
                                                                ("Fuzzing", "Fuzzing"),
                                                                ("License", "License"),
                                                                ("Maintained", "Maintained"),
                                                                ("Packaging", "Packaging"),
                                                                ("Pinned dependencies", "Pinned_Dependencies"),
                                                                ("SAST", "SAST"),
                                                                ("Security policy", "Security_Policy"),
                                                                ("Signed releases", "Signed_Releases"),
                                                                ("Token permissions", "Token_Permissions"),
                                                                ("Vulnerabilities", "Vulnerabilities"),
                                                                ("Total score", "Total_Score"),
                                                            ]
                                                        ],
                                                        style="width: 100%; border-collapse: collapse;",
                                                    )
                                                ),
                                            ),
                                            id="repo_detail_top",
                                        ),
                                        style="border-right: 1px solid #ddd; padding-right: 16px;",
                                    ),
                                    ui.div(
                                        sui.navset_tab(
                                            sui.nav_panel(
                                                "README",
                                                ui.markdown(_readme_md)
                                                if _readme_md
                                                else ui.p("No README available", class_="text-muted"),
                                            ),
                                            sui.nav_panel(
                                                "Contributing",
                                                ui.markdown(_contributing_md)
                                                if _contributing_md
                                                else ui.p("No contributing guide available", class_="text-muted"),
                                            ),
                                            sui.nav_panel(
                                                "Security Policy",
                                                ui.markdown(_security_policy_md)
                                                if _security_policy_md
                                                else ui.p("No security policy available", class_="text-muted"),
                                            ),
                                            id="repo_detail_bottom",
                                        ),
                                        style="padding-left: 16px;",
                                    ),
                                    col_widths=(4, 6),
                                )

                                # ---- Impact ----
                with ui.nav_panel("Impact"):
                        # Table + value boxes (2x2)
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card(fill=True):
                                ui.markdown("**Impact Indicators per University**")
                                @render.data_frame
                                def impact_leaderboard_table():
                                    data = filtered_df()
                                    if data.is_empty():
                                        return render.DataGrid(pl.DataFrame({
                                            "University": [], "Total\nstars": [], "Total\nforks": [],
                                            "Total\ndownloads": [], "Total\ncontributors": [],
                                        }))

                                    uni_expr = (
                                        pl.col("university").fill_null("Unknown")
                                        if "university" in data.columns
                                        else pl.lit("Unknown")
                                    ).alias("_uni")
                                    agg = {
                                        "Total\nstars": "stargazers_count",
                                        "Total\nforks": "forks_count",
                                        "Total\ndownloads": "release_downloads",
                                        "Total\ncontributors": "contributor_count",
                                    }
                                    agg_exprs = [
                                        pl.col(src).cast(pl.Float64, strict=False).sum().alias(dst)
                                        if src in data.columns
                                        else pl.lit(None).cast(pl.Float64).alias(dst)
                                        for dst, src in agg.items()
                                    ]
                                    out = (
                                        data.with_columns(uni_expr)
                                        .group_by("_uni")
                                        .agg(agg_exprs)
                                        .sort("Total\nstars", descending=True, nulls_last=True)
                                        .rename({"_uni": "University"})
                                    )
                                    for col in agg:
                                        out = out.with_columns(
                                            pl.col(col)
                                            .map_elements(_format_thousands_approx, return_dtype=pl.Utf8)
                                            .alias(col)
                                        )

                                    return render.DataGrid(
                                        out,
                                        width="100%",
                                        height="400px",
                                        styles=[
                                            {"location": "body", "style": {"fontSize": _TABLE_FONT_SIZE}},
                                            {"location": "body", "cols": [0], "style": {"minWidth": "32%", "width": "32%"}},
                                            {"location": "body", "cols": [1], "style": {"minWidth": "17%", "width": "17%", "textAlign": "right"}},
                                            {"location": "body", "cols": [2], "style": {"minWidth": "17%", "width": "17%", "textAlign": "right"}},
                                            {"location": "body", "cols": [3], "style": {"minWidth": "17%", "width": "17%", "textAlign": "right"}},
                                            {"location": "body", "cols": [4], "style": {"minWidth": "17%", "width": "17%", "textAlign": "right"}},
                                        ],
                                    )

                            with ui.div():
                                with ui.layout_columns(col_widths=(6, 6)):
                                    with ui.value_box(showcase=ICONS["stars"]):
                                        "Total stars"
                                        @render.express
                                        def impact_total_stars():
                                            data = filtered_df()
                                            if "stargazers_count" not in data.columns:
                                                "—"
                                            else:
                                                s = data["stargazers_count"].cast(pl.Float64, strict=False).drop_nulls()
                                                int(s.sum()) if len(s) > 0 else 0

                                    with ui.value_box(showcase=ICONS["forks"]):
                                        "Total forks"
                                        @render.express
                                        def impact_total_forks():
                                            data = filtered_df()
                                            if "forks_count" not in data.columns:
                                                "—"
                                            else:
                                                s = data["forks_count"].cast(pl.Float64, strict=False).drop_nulls()
                                                int(s.sum()) if len(s) > 0 else 0

                                with ui.layout_columns(col_widths=(6, 6)):
                                    with ui.value_box(showcase=ICONS["downloads"]):
                                        "Total downloads"
                                        @render.express
                                        def impact_total_downloads():
                                            data = filtered_df()
                                            if "release_downloads" not in data.columns:
                                                "—"
                                            else:
                                                s = data["release_downloads"].cast(pl.Float64, strict=False).drop_nulls()
                                                int(s.sum()) if len(s) > 0 else 0

                                    with ui.value_box(showcase=ICONS["contributors"]):
                                        "Total contributors"
                                        @render.express
                                        def impact_total_contributors():
                                            data = filtered_df()
                                            if "contributor_count" not in data.columns:
                                                "—"
                                            else:
                                                s = data["contributor_count"].cast(pl.Float64, strict=False).drop_nulls()
                                                int(s.sum()) if len(s) > 0 else 0

                        # Distribution plots — 2 per row
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_impact_stars():
                                    return plot_stars_distribution_bar_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_impact_forks():
                                    return plot_forks_distribution_bar_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_impact_downloads():
                                    return plot_release_downloads_distribution_bar_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_impact_contributors():
                                    return plot_contributors_distribution_bar_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                                # ---- Sustainability ----
                with ui.nav_panel("Sustainability"):
                        # Row 1: wider table + 2 stacked value boxes
                        with ui.layout_columns(col_widths=(9, 3)):
                            with ui.card():
                                ui.markdown("**Sustainability indicators per University**")
                                @render.data_frame
                                def sustainability_leaderboard_table():
                                    data = filtered_df()
                                    if data.is_empty():
                                        return render.DataGrid(pl.DataFrame({
                                            "University": [], "Average\n# Contributors": [], "Average\nbus factor": [],
                                        }))

                                    uni_expr = (
                                        pl.col("university").fill_null("Unknown")
                                        if "university" in data.columns
                                        else pl.lit("Unknown")
                                    ).alias("_uni")
                                    agg_exprs = [
                                        pl.col("contributor_count").cast(pl.Float64, strict=False).mean().alias("_avg_contrib")
                                        if "contributor_count" in data.columns
                                        else pl.lit(None).cast(pl.Float64).alias("_avg_contrib"),
                                        pl.col("bus_factor").cast(pl.Float64, strict=False).mean().alias("_avg_bus")
                                        if "bus_factor" in data.columns
                                        else pl.lit(None).cast(pl.Float64).alias("_avg_bus"),
                                    ]
                                    agg = (
                                        data.with_columns(uni_expr)
                                        .group_by("_uni")
                                        .agg(agg_exprs)
                                        .sort("_avg_contrib", descending=True, nulls_last=True)
                                    )

                                    def _fmt(v):
                                        return f"{v:.2f}" if v is not None and v == v else "—"

                                    out = agg.with_columns([
                                        pl.col("_avg_contrib").map_elements(_fmt, return_dtype=pl.Utf8).alias("Average\n# Contributors"),
                                        pl.col("_avg_bus").map_elements(_fmt, return_dtype=pl.Utf8).alias("Average\nbus factor"),
                                    ]).rename({"_uni": "University"}).select(["University", "Average\n# Contributors", "Average\nbus factor"])

                                    return render.DataGrid(
                                        out,
                                        width="100%",
                                        height="320px",
                                        styles=[
                                            {"location": "body", "style": {"fontSize": _TABLE_FONT_SIZE}},
                                            {"location": "body", "cols": [0], "style": {"minWidth": "40%", "width": "40%"}},
                                            {"location": "body", "cols": [1], "style": {"minWidth": "30%", "width": "30%", "textAlign": "right"}},
                                            {"location": "body", "cols": [2], "style": {"minWidth": "30%", "width": "30%", "textAlign": "right"}},
                                        ],
                                    )

                            with ui.div():
                                with ui.value_box(showcase=ICONS["busfactor"]):
                                    "Average bus factor"
                                    @render.express
                                    def sustainability_value_avg_bus_factor():
                                        data = filtered_df()
                                        col = "bus_factor"
                                        if col not in data.columns:
                                            "—"
                                        else:
                                            v = data[col].cast(pl.Float64, strict=False).mean()
                                            f"{v:.2f}" if v is not None else "—"

                                with ui.value_box(showcase=ICONS["contributors"]):
                                    "Average # contributors"
                                    @render.express
                                    def sustainability_value_avg_contributors():
                                        data = filtered_df()
                                        if "contributor_count" not in data.columns:
                                            "—"
                                        else:
                                            v = data["contributor_count"].cast(pl.Float64, strict=False).mean()
                                            f"{v:.2f}" if v is not None else "—"

                        # Plots — 2 per row
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_files():
                                    return plot_feature_counts_per_type_altair(
                                        filtered_df(),
                                        FEATURES,
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_heatmap():
                                    return plot_feature_heatmap_by_star_bucket_altair(
                                        filtered_df(),
                                        FEATURES,
                                        star_col="stargazers_count",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        annotations_size=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.card():
                                @render_altair
                                def plot_bus_factor_distribution():
                                    return plot_bus_factor_distribution_bar_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                            with ui.card():
                                @render_altair
                                def plot_contributor_count_buckets():
                                    return plot_contributor_count_bucket_bar_altair(
                                        filtered_df(),
                                        acronym="",
                                        label_size=_OVERVIEW_LABEL_SIZE,
                                        title_size=_OVERVIEW_TITLE_SIZE,
                                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                                    )

                                # ---- Security ----
                with ui.nav_panel("Security"):
                        with ui.layout_columns(col_widths=(8, 4)):
                            with ui.card():
                                ui.markdown(
                                    "**Security scorecard by repository** ([OpenSSF Scorecard](https://scorecard.dev/))"
                                )
                                @render.data_frame
                                def security_scorecard_table():
                                    out = security_repositories_table_df()
                                    if out.is_empty():
                                        return render.DataGrid(out)
                                    return render.DataGrid(
                                        out,
                                        width="100%",
                                        height="650px",
                                        styles=[
                                            {
                                                "location": "body",
                                                "style": {"fontSize": "12px"},
                                            },
                                        ],
                                    )

                            with ui.card():
                                ui.markdown(
                                    "**Average score per Security Metric**"
                                )
                                @render_altair
                                def security_metric_averages_heatmap():
                                    df_avg = security_metric_averages_df()

                                    if df_avg.is_empty() or df_avg["Average"].is_not_null().sum() == 0:
                                        return (
                                            alt.Chart(pl.DataFrame({"Metric": [], "x": [], "Average": []}).to_pandas())
                                            .mark_rect()
                                            .properties(title="Metric averages")
                                        )

                                    df_avg = df_avg.with_columns([
                                        pl.col("Average").map_elements(
                                            lambda v: f"{v:.2f}" if v is not None else "",
                                            return_dtype=pl.Utf8,
                                        ).alias("Label"),
                                        pl.lit("Average").alias("x"),
                                    ])
                                    metric_order = df_avg["Metric"].to_list()

                                    # Exclude spacer row (Metric=" ") from rendering
                                    plot_df = df_avg.filter(pl.col("Metric").str.strip_chars() != "")

                                    rects = (
                                        alt.Chart(plot_df)
                                        .mark_rect(stroke="white", strokeWidth=0.6)
                                        .encode(
                                            x=alt.X(
                                                "x:N",
                                                title="",
                                                axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE),
                                            ),
                                            y=alt.Y(
                                                "Metric:N",
                                                sort=metric_order,
                                                title="",
                                                axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE),
                                            ),
                                            color=alt.Color(
                                                "Average:Q",
                                                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 10]),
                                                legend=None,
                                            ),
                                            tooltip=[
                                                alt.Tooltip("Metric:N"),
                                                alt.Tooltip("Average:Q", title="Average", format=".2f"),
                                            ],
                                        )
                                    )

                                    texts = (
                                        alt.Chart(plot_df)
                                        .mark_text(fontSize=_OVERVIEW_LABEL_SIZE, color="black")
                                        .encode(
                                            x=alt.X("x:N"),
                                            y=alt.Y("Metric:N", sort=metric_order),
                                            text="Label:N",
                                        )
                                    )

                                    return (
                                        (rects + texts)
                                        .properties(
                                            width=alt.Step(120),
                                            height=alt.Step(28),
                                            title="Metric averages",
                                        )
                                        .configure_title(fontSize=_OVERVIEW_TITLE_SIZE, anchor="middle")
                                        .configure_axis(titleFontSize=_OVERVIEW_LABEL_SIZE)
                                        .configure_view(stroke=None)
                                    )

    # ============================= ORGANIZATIONS TAB ==============================================
    with ui.nav_panel("Organizations"):
        with ui.layout_sidebar(fillable=True):
            with ui.sidebar(open="open", bg="#f8f8f8", width="300px"):
                ui.input_slider(
                    "slider_org_threshold",
                    "Prediction Threshold:",
                    min=0.0, max=1.0,
                    value=[0.5, 1.0],
                    step=0.01,
                )
                ui.input_selectize("org_university", "University:", _org_unis, multiple=True)
                ui.br()
                ui.input_action_button("reset_org_filters", "Reset Org Filters", class_="btn-danger")
                ui.HTML("")

            with ui.navset_tab(id="org_tab", selected="Overview"):

                # ---- Overview ----
                with ui.nav_panel("Overview"):
                    with ui.layout_columns(col_widths=(6, 6)):
                        # Value boxes 2x2
                        with ui.layout_columns(col_widths=(6, 6)):
                            with ui.value_box(showcase=ICONS["organizations"]):
                                "Total organizations"
                                @render.express
                                def org_value_total():
                                    f"{len(filtered_org_df()):,}"

                            with ui.value_box(showcase=ICONS["link"]):
                                "With URL"
                                @render.express
                                def org_value_url():
                                    data = filtered_org_df()
                                    if data.is_empty() or "url" not in data.columns:
                                        "—"
                                    else:
                                        has = data.filter(
                                            pl.col("url").is_not_null() & (pl.col("url").str.strip_chars().str.len_chars() > 0)
                                        ).height
                                        f"{has / data.height * 100:.1f}%"

                            with ui.value_box(showcase=ICONS["description"]):
                                "With description"
                                @render.express
                                def org_value_description():
                                    data = filtered_org_df()
                                    if data.is_empty() or "description" not in data.columns:
                                        "—"
                                    else:
                                        has = data.filter(
                                            pl.col("description").is_not_null() & (pl.col("description").str.strip_chars().str.len_chars() > 0)
                                        ).height
                                        f"{has / data.height * 100:.1f}%"

                            with ui.value_box(showcase=ICONS["email"]):
                                "With email"
                                @render.express
                                def org_value_email():
                                    data = filtered_org_df()
                                    if data.is_empty() or "email" not in data.columns:
                                        "—"
                                    else:
                                        has = data.filter(
                                            pl.col("email").is_not_null() & (pl.col("email").str.strip_chars().str.len_chars() > 0)
                                        ).height
                                        f"{has / data.height * 100:.1f}%"

                        # Organizations per university chart
                        with ui.card():
                            ui.markdown("**Organizations per university**")
                            @render_altair
                            def org_plot_per_university():
                                data = filtered_org_df()
                                if data.is_empty() or "university" not in data.columns:
                                    return alt.Chart(pl.DataFrame({"university": [], "count": []}).to_pandas()).mark_bar().properties(width="container", height="container")
                                counts = (
                                    data.group_by("university")
                                    .agg(pl.len().alias("count"))
                                    .sort("count", descending=True)
                                    .head(20)
                                ).to_pandas()
                                return (
                                    alt.Chart(counts)
                                    .mark_bar(color="#378ADD")
                                    .encode(
                                        x=alt.X("count:Q", title="Organizations", axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE)),
                                        y=alt.Y("university:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE)),
                                        tooltip=[alt.Tooltip("university:N", title="University"), alt.Tooltip("count:Q", title="Count")],
                                    )
                                    .properties(width="container", height="container", title="")
                                    .configure_view(stroke=None)
                                )

                    with ui.layout_columns(col_widths=(6, 6)):
                        with ui.card():
                            ui.markdown("**Organizations created per year**")
                            @render_altair
                            def org_plot_created_per_year():
                                data = filtered_org_df()
                                if data.is_empty() or "created_at" not in data.columns:
                                    return alt.Chart(pl.DataFrame({"year": [], "count": []}).to_pandas()).mark_bar().properties(width="container", height="container")
                                counts = (
                                    data.with_columns(
                                        pl.col("created_at").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
                                    )
                                    .filter(pl.col("year").is_not_null())
                                    .group_by("year")
                                    .agg(pl.len().alias("count"))
                                    .sort("year")
                                ).to_pandas()
                                return (
                                    alt.Chart(counts)
                                    .mark_bar(color="#185FA5")
                                    .encode(
                                        x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=-45, labelFontSize=_OVERVIEW_LABEL_SIZE)),
                                        y=alt.Y("count:Q", title="Organizations", axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE)),
                                        tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("count:Q", title="Count")],
                                    )
                                    .properties(width="container", height="container", title="")
                                    .configure_view(stroke=None)
                                )

                        with ui.card():
                            ui.markdown("**Profile completeness**")
                            @render_altair
                            def org_plot_profile_completeness():
                                data = filtered_org_df()
                                if data.is_empty():
                                    return alt.Chart(pl.DataFrame({"field": [], "pct": []}).to_pandas()).mark_bar().properties(width="container", height="container")
                                fields = {
                                    "Description": "description",
                                    "Location": "location",
                                    "Website": "url",
                                    "Email": "email",
                                    "Company": "company",
                                }
                                rows = []
                                for label, col in fields.items():
                                    if col in data.columns:
                                        filled = data.filter(
                                            pl.col(col).is_not_null() & (pl.col(col).str.strip_chars().str.len_chars() > 0)
                                        ).height
                                        pct = round(filled / data.height * 100, 1)
                                    else:
                                        pct = 0.0
                                    rows.append({"field": label, "pct": pct})
                                df_plot = pl.DataFrame(rows).to_pandas()
                                return (
                                    alt.Chart(df_plot)
                                    .mark_bar(color="#1D9E75")
                                    .encode(
                                        x=alt.X("pct:Q", title="% filled", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE)),
                                        y=alt.Y("field:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=_OVERVIEW_LABEL_SIZE)),
                                        tooltip=[alt.Tooltip("field:N", title="Field"), alt.Tooltip("pct:Q", title="% filled")],
                                    )
                                    .properties(width="container", height="container", title="")
                                    .configure_view(stroke=None)
                                )

                # ---- Browse ----
                with ui.nav_panel("Browse"):
                    with ui.card(class_="repo-data-card"):
                        ui.tags.div(
                            ui.tags.div(
                                ui.input_text(
                                    "org_search",
                                    "Search",
                                    placeholder="Search organizations...",
                                    width="100%",
                                ),
                                style="flex: 1; min-width: 220px;",
                            ),
                        )
                        @render.data_frame
                        def org_browse_table():
                            data = filtered_org_df()
                            if data.is_empty():
                                return render.DataGrid(pl.DataFrame())

                            search = (input.org_search() or "").strip().lower()
                            if search:
                                search_cols = ["login", "name", "description", "university", "location", "email", "url"]
                                conditions = [
                                    pl.col(c).cast(pl.Utf8).str.to_lowercase().str.contains(search, literal=True)
                                    for c in search_cols if c in data.columns
                                ]
                                if conditions:
                                    data = data.filter(pl.any_horizontal(conditions))

                            col_map = {
                                "login": "Login",
                                "name": "Name",
                                "university": "University",
                                "description": "Description",
                                "company": "Company",
                                "email": "Email",
                                "url": "URL",
                                "location": "Location",
                                "source": "Source",
                                "created_at": "Created",
                                "affiliation_prediction_orgs": "Affiliation score",
                            }
                            display_cols = [c for c in col_map if c in data.columns]
                            out = data.select(display_cols).rename(col_map)

                            if "Created" in out.columns:
                                out = out.with_columns(
                                    pl.col("Created").str.slice(0, 7).alias("Created")
                                )
                            if "Affiliation score" in out.columns:
                                out = out.with_columns(
                                    pl.col("Affiliation score")
                                    .cast(pl.Float64, strict=False)
                                    .map_elements(lambda v: f"{v:.2f}" if v is not None else "—", return_dtype=pl.Utf8)
                                    .alias("Affiliation score")
                                )
                            if "Login" in out.columns:
                                login_idx = out.columns.index("Login")
                                github_url = out["Login"].map_elements(
                                    lambda v: f"https://github.com/{v}" if v and str(v).strip() else "—",
                                    return_dtype=pl.Utf8,
                                ).alias("GitHub URL")
                                cols_before = out.columns[:login_idx + 1]
                                cols_after = out.columns[login_idx + 1:]
                                out = out.select(list(cols_before) + [github_url] + [pl.col(c) for c in cols_after])

                            return render.DataGrid(
                                out,
                                height="500px",
                                selection_mode="row",
                            )

                        @render.download(
                            filename=lambda: f"organizations_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )
                        def download_orgs_csv():
                            data = filtered_org_df()
                            buf = io.BytesIO()
                            data.write_csv(buf)
                            buf.seek(0)
                            yield buf.getvalue()


# Reset org filters
@reactive.effect
@reactive.event(input.reset_org_filters)
def _reset_org_filters():
    ui.update_selectize("org_university", selected=[])
    ui.update_slider("slider_org_threshold", value=[0.5, 1.0])

# ------------------------------------ Filtered DataFrame ----------------------------------------------

@reactive.calc
def filtered_org_df():
    data = df_organizations
    if data.is_empty():
        return data

    if input.org_university():
        data = data.filter(pl.col("university").is_in(list(input.org_university())))
    if "affiliation_prediction_orgs" in data.columns and input.slider_org_threshold():
        min_val, max_val = input.slider_org_threshold()
        data = data.filter(pl.col("affiliation_prediction_orgs").is_between(min_val, max_val))

    return data


@reactive.calc
def filtered_df():
    result = df

    if input.university():
        result = result.filter(pl.col("university").is_in(list(input.university())))
    if input.type():
        result = result.filter(pl.col("type_prediction_gpt_5_mini").is_in(list(input.type())))
    if input.license():
        result = result.filter(pl.col("license").is_in(list(input.license())))
    if input.language():
        result = result.filter(pl.col("language").is_in(list(input.language())))
    if input.slider_stars():
        min_val, max_val = input.slider_stars()
        result = result.filter(pl.col("stargazers_count").is_between(min_val, max_val))
    if input.slider_forks():
        min_val, max_val = input.slider_forks()
        result = result.filter(pl.col("forks_count").is_between(min_val, max_val))
    if input.slider_downloads():
        min_val, max_val = input.slider_downloads()
        result = result.filter(pl.col("release_downloads").is_between(min_val, max_val))
    if input.slider_threshold():
        min_val, max_val = input.slider_threshold()
        result = result.filter(pl.col("affiliation_prediction_gpt_5_mini").is_between(min_val, max_val))

    # Chat filter (ENABLE_CHAT=False; if re-enabled, querychat must be updated for Polars)
    if ENABLE_CHAT and chat is not None:
        try:
            chat_df = chat.df()
            if chat_df is not None and len(chat_df) > 0:
                if "id" in chat_df.columns:
                    chat_ids = list(chat_df["id"].values)
                    result = result.filter(pl.col("id").is_in(chat_ids))
        except Exception:
            pass

    return result


_REPO_TABLE_DROP_COLS = [
    "readme",
    "contributing",
    "contributors",
    "code_of_conduct_file",
    "security_policy",
    "issue_templates",
    "pull_request_template",
]


@reactive.calc
def repositories_table_df():
    """Same rows/columns as the Repositories DataGrid (filters + search)."""
    drop = [c for c in _REPO_TABLE_DROP_COLS if c in filtered_df().columns]
    data = filtered_df().drop(drop)
    search_term = (input.table_search() or "").strip().lower()
    if search_term:
        searchable_columns = [
            "full_name", "owner", "description", "language", "license",
            "university", "affiliation_prediction_gpt_5_mini",
        ]
        conditions = [
            pl.col(c).cast(pl.Utf8).str.to_lowercase().str.contains(search_term, literal=True)
            for c in searchable_columns if c in data.columns
        ]
        if conditions:
            data = data.filter(pl.any_horizontal(conditions))
    return data


@reactive.calc
def security_repositories_table_df():
    """
    One row per filtered repository: ``html_url`` plus scorecard columns from
    ``df_security`` (left join on ``html_url``).
    """
    base = filtered_df()
    work = (
        base.select("html_url") if "html_url" in base.columns
        else base.select([]).with_columns(pl.lit(None).cast(pl.Utf8).alias("html_url"))
    )

    metric_pairs = [
        (d, s) for d, s in SECURITY_SCORECARD_METRICS if s in df_security.columns
    ]
    can_merge = (
        not df_security.is_empty()
        and "html_url" in df_security.columns
        and bool(metric_pairs)
    )

    if can_merge:
        s_cols = ["html_url"] + [s for _, s in metric_pairs]
        sec = (
            df_security.select(s_cols)
            .unique(subset=["html_url"], keep="first")
            .rename({s: d for d, s in metric_pairs})
        )
        out = work.join(sec, on="html_url", how="left")
    else:
        out = work

    for d, _ in SECURITY_SCORECARD_METRICS:
        if d not in out.columns:
            out = out.with_columns(pl.lit(None).alias(d))

    _total_col = "Total score"
    if _total_col in out.columns:
        out = (
            out.with_columns(pl.col(_total_col).cast(pl.Float64, strict=False).alias("_sort_key"))
            .sort("_sort_key", descending=True, nulls_last=True)
            .drop("_sort_key")
        )

    metric_displays = [d for d, _ in SECURITY_SCORECARD_METRICS]
    return out.select(["html_url"] + [d for d in metric_displays if d in out.columns])


@reactive.calc
def security_metric_averages_df():
    """
    One row per scorecard metric: mean of numeric values, excluding −1 and non-finite.
    """
    wide = security_repositories_table_df()
    rows = []
    for disp, _src in SECURITY_SCORECARD_METRICS:
        if disp not in wide.columns:
            rows.append({"Metric": disp, "Average": None})
            continue
        s = wide[disp].cast(pl.Float64, strict=False)
        valid = s.filter(s.is_not_null() & (s != -1))
        avg = float(valid.mean()) if len(valid) > 0 else None
        rows.append({"Metric": disp, "Average": avg})
    out = pl.DataFrame(rows, schema={"Metric": pl.Utf8, "Average": pl.Float64})
    total_mask = out["Metric"] == "Total score"
    main = out.filter(~total_mask).sort("Average", descending=True, nulls_last=True)
    total_row = out.filter(total_mask)
    if total_row.is_empty():
        return main
    sep_row = pl.DataFrame({"Metric": [" "], "Average": [None]}, schema={"Metric": pl.Utf8, "Average": pl.Float64})
    return pl.concat([main, sep_row, total_row])

