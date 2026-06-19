#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
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
from repoexplorer.analysis.stars_distribution_bar import (
    plot_stars_distribution_bar,
    plot_stars_distribution_bar_altair,
)
from repoexplorer.analysis.forks_distribution_bar import (
    plot_forks_distribution_bar,
    plot_forks_distribution_bar_altair,
)
from repoexplorer.analysis.release_downloads_distribution_bar import (
    plot_release_downloads_distribution_bar,
    plot_release_downloads_distribution_bar_altair,
)
from repoexplorer.analysis.contributors_distribution_bar import (
    plot_contributors_distribution_bar,
    plot_contributors_distribution_bar_altair,
)
from repoexplorer.analysis.bus_factor_distribution_bar import (
    plot_bus_factor_distribution_bar,
    plot_bus_factor_distribution_bar_altair,
)
from repoexplorer.analysis.contributor_count_bucket_bar import (
    plot_contributor_count_bucket_bar,
    plot_contributor_count_bucket_bar_altair,
)
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
# ORGANIZATIONS_PARQUET = os.path.join(PARQUET_BASE, "organizations_combined_clean.parquet")
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
    """True for None, NaN, or pandas NA (common for null cells from parquet)."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
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
    if pd.isna(x):
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


def _normalize_license_column(df: pd.DataFrame) -> None:
    """
    In-place: missing / empty licenses become pandas NA.

    Remote parquet often stores null licenses as real NaN. Using ``.astype(str)``
    in metrics turns those into the literal "nan" string, which incorrectly counts
    as having a license (~100%). Local files may use empty strings instead.
    """
    if df is None or df.empty or "license" not in df.columns:
        return
    s = df["license"].astype("string").str.strip()
    no_license = s.isna() | (s.str.lower().isin(["", "none", "nan", "null", "<na>"]))
    df["license"] = s.mask(no_license, pd.NA)


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
    return pd.read_parquet(url, columns=columns)

def optimize_dtypes(df: pd.DataFrame):
    for col in ["stargazers_count", "forks_count", "watchers_count",
                "open_issues_count", "subscribers_count", "contributor_count",
                "bus_factor", "release_downloads"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    if "affiliation_prediction_gpt_5_mini" in df.columns:
        df["affiliation_prediction_gpt_5_mini"] = pd.to_numeric(
            df["affiliation_prediction_gpt_5_mini"], errors="coerce"
        ).astype("float32")
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
    df = read_parquet_from_s3_public("repoexplorer-data", "repositories_reduced_combined_stars_gt_0.parquet", columns=COLUMNS_TO_LOAD)
    df_security = read_parquet_from_s3_public("repoexplorer-data", "security_reduced_combined_stars_gt_0.parquet")

else:
    # Load main repositories table
    cload = COLUMNS_TO_LOAD
    df = pd.read_parquet(COMBINED_PARQUET, columns=cload)
    if "university" not in df.columns:
        df["university"] = "Unknown"
    df_security = pd.DataFrame()
    if os.path.isfile(SECURITY_PARQUET):
        try:
            df_security = pd.read_parquet(SECURITY_PARQUET)
        except Exception:
            logging.exception("Failed to load security parquet %s", SECURITY_PARQUET)
            df_security = pd.DataFrame()

df = optimize_dtypes(df)


# # Load organizations table
# df_organizations = pd.DataFrame()
# if os.path.isfile(ORGANIZATIONS_PARQUET):
#     try:
#         df_organizations = pd.read_parquet(ORGANIZATIONS_PARQUET)
#     except Exception:
#         logging.exception("Failed to load organizations parquet %s", ORGANIZATIONS_PARQUET)
#         df_organizations = pd.DataFrame()

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


_normalize_license_column(df)


# =============================================== App UI ==========================================

ui.page_opts(title="Open Source Repository Browser", fillable=True)

# ======================================== Sidebar  ===============================================

# ------------------------------------ Manual Filters ----------------------------------------------

licenses = df["license"].dropna().unique().tolist() if "license" in df.columns else []
languages = df["language"].unique().tolist() if "language" in df.columns else []
universities = df["university"].unique().tolist() if "university" in df.columns else []
types = df["type_prediction_gpt_5_mini"].unique().tolist() if "type_prediction_gpt_5_mini" in df.columns else []

# Subset with default prediction threshold (>= 0.8) for sliders and for chat
_df_aff = pd.to_numeric(df["affiliation_prediction_gpt_5_mini"], errors="coerce")
_df_08 = df.loc[_df_aff >= 0.8]
_m = pd.to_numeric(_df_08["stargazers_count"], errors="coerce").max() if not _df_08.empty else None
_slider_max_stars = int(_m) if _m is not None and not pd.isna(_m) else 5000
_m = pd.to_numeric(_df_08["forks_count"], errors="coerce").max() if not _df_08.empty else None
_slider_max_forks = int(_m) if _m is not None and not pd.isna(_m) else 100
_m = pd.to_numeric(_df_08["release_downloads"], errors="coerce").max() if not _df_08.empty else None
_slider_max_downloads = int(_m) if _m is not None and not pd.isna(_m) else 1000

# ------------------------------------ QueryChat Config -------------------------------------------
# Chat uses only repos with prediction threshold >= 0.8 (same as default slider)
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

with ui.sidebar(open="open", bg="#f8f8f8", width="300px"):  
   with ui.navset_pill(id="side_tab"): 
       with ui.nav_panel("Manual Filters"):
           max_threshold = pd.to_numeric(df["affiliation_prediction_gpt_5_mini"], errors="coerce").max()
           ui.input_slider("slider_threshold", "Prediction Threshold", min=0, max=1, value=[0.8, 1])  
           
           ui.input_selectize(  
                "university",  
                "University:",  
                universities,
                multiple=True,  
            ) 
           ui.input_selectize(  
                "type",  
                "Project Type:",  
                types,
                multiple=True,  
            )  
           ui.input_selectize(  
                "license",  
                "License:",  
                licenses,
                multiple=True,  
            )     
           ui.input_selectize(  
                "language",  
                "Language:",  
                languages,
                multiple=True,  
            )  
           ui.input_slider("slider_stars", "# Stars", min=0, max=_slider_max_stars, value=[0, _slider_max_stars])
           ui.input_slider("slider_forks", "# Forks", min=0, max=_slider_max_forks, value=[0, _slider_max_forks])
           ui.input_slider("slider_downloads", "# Release Downloads", min=0, max=_slider_max_downloads, value=[0, _slider_max_downloads])  
            
            

       if ENABLE_CHAT:
           with ui.nav_panel("Chat Bot"):
               qc.ui("chat")  # may return True; do not use as last expression
   
   # Reset button at the bottom of the sidebar
   ui.br()
   ui.br()
   ui.input_action_button("reset_filters", "Reset All Filters", class_="btn-danger")
   ui.HTML("")  # prevent any prior return value (e.g. True) from rendering

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
    # Reset manual filter inputs
    ui.update_selectize("university", selected=[])
    ui.update_selectize("type", selected=[])
    ui.update_selectize("license", selected=[])
    ui.update_selectize("language", selected=[])
    ui.update_slider("slider_stars", value=[0, _slider_max_stars])
    ui.update_slider("slider_forks", value=[0, _slider_max_forks])
    ui.update_slider("slider_downloads", value=[0, _slider_max_downloads])
    ui.update_text("table_search", value="")

    # Ask the chatbot to reset its filters (sends "reset all filters" so the LLM can invoke reset_dashboard)
    if ENABLE_CHAT:
        try:
            sess = shiny_session.get_current_session()
            if sess is not None:
                # Send message to chat input so the bot receives "reset all filters" and can clear filters
                sess.send_input_message("chat-message", {"value": "reset all filters"})
        except Exception:
            pass

# ======================================== Main panel  ===============================================

#------------------------------------ Overview ----------------------------------------------

# Typography for Overview tab charts (titles, axis labels, legends, annotations)
_OVERVIEW_TITLE_SIZE = 16
_OVERVIEW_LABEL_SIZE = 14
_OVERVIEW_TEXT_SIZE = 11
_OVERVIEW_BAR_PCT_SIZE = 14
_OVERVIEW_PIE_PCT_SIZE = 14


# Icons for Overview / Impact value boxes
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
}

_about_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "about.md")
with open(_about_path, encoding="utf-8") as _f:
    _about_md = _f.read()

with ui.navset_pill(id="tab", selected="Overview"):
    with ui.nav_panel("About"):
        with ui.card():
            ui.markdown(_about_md)

    with ui.nav_panel("Overview"):
        # Value boxes row
        with ui.layout_columns(col_widths=(3, 3, 3, 3)):
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
                    # Use numeric contributor_count column if available
                    if "contributor_count" not in data.columns:
                        "—"
                    else:
                        counts = pd.to_numeric(data["contributor_count"], errors="coerce")
                        total = int(counts.dropna().sum()) if counts.notna().any() else 0
                        total

            with ui.value_box(showcase=ICONS["license"]):
                "Repositories with a license"
                @render.express
                def pct_with_license():
                    data = filtered_df()
                    if "license" not in data.columns:
                        "—"
                    else:
                        total = len(data)
                        if total == 0:
                            "0%"
                        else:
                            # After _normalize_license_column, null means no license.
                            with_license = int(data["license"].notna().sum())
                            pct = 100.0 * with_license / total
                            f"{pct:.1f}%"

            with ui.value_box(showcase=ICONS["busfactor"]):
                "Average bus factor"
                @render.express
                def avg_busfactor():
                    data = filtered_df()
                    col = "bus_factor"
                    # Compute across the full repositories table (not just filtered_df)
                    if col not in data.columns:
                        "—"
                    else:
                        try:
                            v = pd.to_numeric(data[col], errors="coerce").mean()
                            f"{v:.1f}" if not pd.isna(v) else "—"
                        except Exception:
                            "—"

        # University distribution
        with ui.layout_columns(col_widths=(5, 7)):
            
            with ui.card():
                @render.data_frame
                def university_table():
                    data = filtered_df()
                    if "university" not in data.columns or data.empty:
                        return render.DataGrid(pd.DataFrame(columns=["University", "Count"]))

                    # Aggregate repositories per university and feed into a DataGrid.
                    university_counts = (
                        data["university"]
                        .value_counts()
                        .sort_values(ascending=False)
                        .rename_axis("University")
                        .reset_index(name="Count")
                    )
                    return render.DataGrid(
                        university_counts,
                        width="100%",
                        height="400px",
                        styles=[
                            {
                                "location": "body",
                                "cols": [0],  # University name column
                                "style": {"minWidth": "70%", "width": "70%"},
                            },
                            {
                                "location": "body",
                                "cols": [1],  # Count column
                                "style": {"minWidth": "30%", "width": "30%", "textAlign": "right"},
                            },
                        ],
                    )
            
            with ui.card(height="450px"):
                @render_altair
                def plot_files_combined():
                    chart = plot_feature_counts_altair(
                        filtered_df(),
                        FEATURES,
                        acronym="",
                        label_size=_OVERVIEW_LABEL_SIZE,
                        title_size=_OVERVIEW_TITLE_SIZE,
                        textprops=_OVERVIEW_BAR_PCT_SIZE,
                    )
                    return chart

        with ui.layout_columns(col_widths=(4, 4, 4)):
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
    

#------------------------------------ Repositories ----------------------------------------------
    
    
    with ui.nav_panel("Repositories"):
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
                filename=lambda: f"repositories_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            def download_repositories_csv():
                out_df = repositories_table_df()
                buf = io.BytesIO()
                out_df.to_csv(buf, index=False, encoding="utf-8")
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
                sel = filtered_df().loc[view.index[row_pos]]
                selected = sel.iloc[0] if isinstance(sel, pd.DataFrame) else sel

                _readme_md = _safe_markdown_text(selected.get("readme"))
                _contributing_md = _safe_markdown_text(selected.get("contributing"))
                _security_policy_md = _safe_markdown_text(selected.get("security_policy"))

                # Match security metrics row (from security_combined_clean.parquet) by html_url
                sec_row = None
                if not df_security.empty and "html_url" in df_security.columns:
                    _matches = df_security.loc[df_security["html_url"] == selected.get("html_url")]
                    if not _matches.empty:
                        sec_row = _matches.iloc[0]

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

    # ------------------------------------ Impact (bucket distributions) ------------------------------
    with ui.nav_panel("Impact"):
        with ui.layout_columns(col_widths=(3, 3, 3, 3)):
            with ui.value_box(showcase=ICONS["stars"]):
                "Total stars"
                @render.express #??
                def impact_total_stars():
                    data = filtered_df()
                    if "stargazers_count" not in data.columns:
                        "—"
                    else:
                        s = pd.to_numeric(data["stargazers_count"], errors="coerce")
                        int(s.dropna().sum()) if s.notna().any() else 0

            with ui.value_box(showcase=ICONS["forks"]):
                "Total forks"
                @render.express
                def impact_total_forks():
                    data = filtered_df()
                    if "forks_count" not in data.columns:
                        "—"
                    else:
                        s = pd.to_numeric(data["forks_count"], errors="coerce")
                        int(s.dropna().sum()) if s.notna().any() else 0

            with ui.value_box(showcase=ICONS["downloads"]):
                "Total downloads"
                @render.express
                def impact_total_downloads():
                    data = filtered_df()
                    if "release_downloads" not in data.columns:
                        "—"
                    else:
                        s = pd.to_numeric(data["release_downloads"], errors="coerce")
                        int(s.dropna().sum()) if s.notna().any() else 0

            with ui.value_box(showcase=ICONS["contributors"]):
                "Total contributors"
                @render.express
                def impact_total_contributors():
                    data = filtered_df()
                    if "contributor_count" not in data.columns:
                        "—"
                    else:
                        s = pd.to_numeric(data["contributor_count"], errors="coerce")
                        int(s.dropna().sum()) if s.notna().any() else 0

        with ui.layout_columns(col_widths=(5, 7)):
            with ui.card():
                ui.markdown(
                    "**Impact Indicators per University**"
                )
                @render.data_frame
                def impact_leaderboard_table():
                    data = filtered_df()
                    _cols = [
                        "University",
                        "Total stars",
                        "Total forks",
                        "Total downloads",
                        "Total contributors",
                    ]
                    if data.empty:
                        return render.DataGrid(pd.DataFrame(columns=_cols))

                    work = data.copy()
                    if "university" in work.columns:
                        work["_uni"] = work["university"].fillna("Unknown").astype(str)
                    else:
                        work["_uni"] = "Unknown"

                    rows = []
                    #groupby(...).agg('sum')
                    for uni, grp in work.groupby("_uni", dropna=False):
                        stars = (
                            pd.to_numeric(grp["stargazers_count"], errors="coerce")
                            if "stargazers_count" in grp.columns
                            else pd.Series(dtype=float)
                        ).sum()
                        forks = (
                            pd.to_numeric(grp["forks_count"], errors="coerce")
                            if "forks_count" in grp.columns
                            else pd.Series(dtype=float)
                        ).sum()
                        downloads = (
                            pd.to_numeric(grp["release_downloads"], errors="coerce")
                            if "release_downloads" in grp.columns
                            else pd.Series(dtype=float)
                        ).sum()
                        contributors = (
                            pd.to_numeric(grp["contributor_count"], errors="coerce")
                            if "contributor_count" in grp.columns
                            else pd.Series(dtype=float)
                        ).sum()
                        rows.append((uni, stars, forks, downloads, contributors))

                    out = pd.DataFrame(
                        rows,
                        columns=[
                            "University",
                            "Total stars",
                            "Total forks",
                            "Total downloads",
                            "Total contributors",
                        ],
                    )
                    out["Total stars"] = out["Total stars"].fillna(0)
                    out["Total forks"] = out["Total forks"].fillna(0)
                    out["Total downloads"] = out["Total downloads"].fillna(0)
                    out["Total contributors"] = out["Total contributors"].fillna(0)
                    out = out.sort_values("Total stars", ascending=False)
                    for c in (
                        "Total stars",
                        "Total forks",
                        "Total downloads",
                        "Total contributors",
                    ):
                        out[c] = out[c].map(_format_thousands_approx)

                    return render.DataGrid(
                        out,
                        width="100%",
                        styles=[
                            {
                                "location": "body",
                                "style": {"fontSize": "12px"},
                            },
                            {
                                "location": "body",
                                "cols": [0],
                                "style": {"minWidth": "32%", "width": "32%"},
                            },
                            {
                                "location": "body",
                                "cols": [1],
                                "style": {
                                    "minWidth": "17%",
                                    "width": "17%",
                                    "textAlign": "right",
                                },
                            },
                            {
                                "location": "body",
                                "cols": [2],
                                "style": {
                                    "minWidth": "17%",
                                    "width": "17%",
                                    "textAlign": "right",
                                },
                            },
                            {
                                "location": "body",
                                "cols": [3],
                                "style": {
                                    "minWidth": "17%",
                                    "width": "17%",
                                    "textAlign": "right",
                                },
                            },
                            {
                                "location": "body",
                                "cols": [4],
                                "style": {
                                    "minWidth": "17%",
                                    "width": "17%",
                                    "textAlign": "right",
                                },
                            },
                        ],
                    )

            with ui.div():
                with ui.layout_columns():
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

                with ui.layout_columns():
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

    # ------------------------------------ Sustainability ----------------------------------------------
    with ui.nav_panel("Sustainability"):
        with ui.layout_columns(col_widths=(4, 8)):
            with ui.card():
                ui.markdown(
                    "**Sustainability indicators per University**"
                )
                @render.data_frame
                def sustainability_leaderboard_table():
                    data = filtered_df()
                    _cols = [
                        "University",
                        "Average # Contributors",
                        "Average bus factor",
                    ]
                    if data.empty:
                        return render.DataGrid(pd.DataFrame(columns=_cols))

                    work = data.copy()
                    if "university" in work.columns:
                        work["_uni"] = work["university"].fillna("Unknown").astype(str)
                    else:
                        work["_uni"] = "Unknown"

                    rows = []
                    for uni, grp in work.groupby("_uni", dropna=False):
                        cc = (
                            pd.to_numeric(grp["contributor_count"], errors="coerce")
                            if "contributor_count" in grp.columns
                            else pd.Series(dtype=float)
                        )
                        bf = (
                            pd.to_numeric(grp["bus_factor"], errors="coerce")
                            if "bus_factor" in grp.columns
                            else pd.Series(dtype=float)
                        )
                        avg_c = cc.mean() if cc.notna().any() else float("nan")
                        avg_b = bf.mean() if bf.notna().any() else float("nan")
                        rows.append((uni, avg_c, avg_b))

                    out = pd.DataFrame(
                        rows,
                        columns=["University", "_avg_contrib", "_avg_bus"],
                    )
                    out = out.sort_values(
                        "_avg_contrib", ascending=False, na_position="last"
                    )
                    out["Average # Contributors"] = out["_avg_contrib"].map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )
                    out["Average bus factor"] = out["_avg_bus"].map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )
                    out = out[
                        ["University", "Average # Contributors", "Average bus factor"]
                    ]

                    return render.DataGrid(
                        out,
                        width="100%",
                        styles=[
                            {
                                "location": "body",
                                "style": {"fontSize": "12px"},
                            },
                            {
                                "location": "body",
                                "cols": [0],
                                "style": {"minWidth": "40%", "width": "40%"},
                            },
                            {
                                "location": "body",
                                "cols": [1],
                                "style": {
                                    "minWidth": "30%",
                                    "width": "30%",
                                    "textAlign": "right",
                                },
                            },
                            {
                                "location": "body",
                                "cols": [2],
                                "style": {
                                    "minWidth": "30%",
                                    "width": "30%",
                                    "textAlign": "right",
                                },
                            },
                        ],
                    )

            with ui.div():
                with ui.layout_columns(col_widths=(6, 6)):
                    with ui.value_box(showcase=ICONS["busfactor"]):
                        "Average bus factor"
                        @render.express
                        def sustainability_value_avg_bus_factor():
                            data = filtered_df()
                            col = "bus_factor"
                            if col not in data.columns:
                                "—"
                            else:
                                v = pd.to_numeric(data[col], errors="coerce").mean()
                                f"{v:.2f}" if not pd.isna(v) else "—"

                    with ui.value_box(showcase=ICONS["contributors"]):
                        "Average # contributors"
                        @render.express
                        def sustainability_value_avg_contributors():
                            data = filtered_df()
                            if "contributor_count" not in data.columns:
                                "—"
                            else:
                                v = pd.to_numeric(
                                    data["contributor_count"], errors="coerce"
                                ).mean()
                                f"{v:.2f}" if not pd.isna(v) else "—"

                with ui.layout_columns():
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

                with ui.layout_columns():
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

    # ------------------------------------ Security (scorecard table) --------------------------------
    with ui.nav_panel("Security"):
        with ui.layout_columns(col_widths=(8, 4)):
            with ui.card():
                ui.markdown(
                    "**Security scorecard by repository** ([OpenSSF Scorecard](https://scorecard.dev/))"
                )
                @render.data_frame
                def security_scorecard_table():
                    out = security_repositories_table_df()
                    if out.empty:
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
                @render.plot
                def security_metric_averages_heatmap():
                    df_avg = security_metric_averages_df()
                    n = len(df_avg)
                    fig_h = max(7.0, 0.38 * n + 1.2)
                    # Narrow width → less-wide “Average” cells (heatmap column)
                    fig, ax = plt.subplots(figsize=(3.5, fig_h))
                    if df_avg.empty or df_avg["Average"].notna().sum() == 0:
                        ax.text(
                            0.5,
                            0.5,
                            "No numeric scores to average\n(after excluding −1)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                        ax.set_axis_off()
                        return fig

                    plot_mat = df_avg.set_index("Metric")[["Average"]]
                    _ann = []
                    for metric, row in plot_mat.iterrows():
                        v = row["Average"]
                        if str(metric).strip() == "":
                            _ann.append("")
                        elif pd.notna(v):
                            _ann.append(f"{v:.2f}")
                        else:
                            _ann.append("—")
                    annot_mat = pd.DataFrame(
                        _ann, index=plot_mat.index, columns=["Average"]
                    )
                    cmap = plt.colormaps["RdYlGn"]
                    mask = plot_mat.isna()
                    sns.heatmap(
                        plot_mat,
                        annot=annot_mat,
                        fmt="",
                        cmap=cmap,
                        vmin=0,
                        vmax=10,
                        mask=mask,
                        linewidths=0.6,
                        linecolor="white",
                        ax=ax,
                        cbar=False,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_title("Metric averages")
                    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
                    plt.setp(ax.get_xticklabels(), fontsize=8)
                    # Spacer row adds gap before “Total score”; hide its tick/label (no empty tick).
                    for i, lab in enumerate(ax.get_yticklabels()):
                        if not lab.get_text().strip():
                            lab.set_visible(False)
                            yticks = ax.yaxis.get_major_ticks()
                            if i < len(yticks):
                                yticks[i].tick1line.set_visible(False)
                                yticks[i].tick2line.set_visible(False)
                    fig.tight_layout()
                    return fig

# ------------------------------------ Filtered DataFrame ----------------------------------------------

@reactive.calc
def filtered_df():
    mask = pd.Series(True, index=df.index)

    if input.university():
        mask &= df["university"].isin(input.university())
    if input.type():
        mask &= df["type_prediction_gpt_5_mini"].isin(input.type())
    if input.license():
        mask &= df["license"].isin(input.license())
    if input.language():
        mask &= df["language"].isin(input.language())
    if input.slider_stars():
        min_val, max_val = input.slider_stars()
        stars = pd.to_numeric(df["stargazers_count"], errors="coerce")
        mask &= (stars >= min_val) & (stars <= max_val)
    if input.slider_forks():
        min_val, max_val = input.slider_forks()
        forks = pd.to_numeric(df["forks_count"], errors="coerce")
        mask &= (forks >= min_val) & (forks <= max_val)
    if input.slider_downloads():
        min_val, max_val = input.slider_downloads()
        downloads = pd.to_numeric(df["release_downloads"], errors="coerce")
        mask &= (downloads >= min_val) & (downloads <= max_val)
    if input.slider_threshold():
        min_val, max_val = input.slider_threshold()
        aff = pd.to_numeric(df["affiliation_prediction_gpt_5_mini"], errors="coerce")
        mask &= (aff >= min_val) & (aff <= max_val)

    # Apply chat filters - combine with manual filters (only when chat is active and has results)
    result = df.loc[mask]

    if ENABLE_CHAT and chat is not None:
        try:
            chat_df = chat.df()
            if chat_df is not None and len(chat_df) > 0:
                if "id" in chat_df.columns:
                    chat_ids = set(chat_df["id"].values)
                    result = result[result["id"].isin(chat_ids)]
                else:
                    chat_indices = set(chat_df.index)
                    result = result[result.index.isin(chat_indices)]
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
    data = filtered_df().drop(columns=_REPO_TABLE_DROP_COLS, errors="ignore")
    if input.table_search() and len(input.table_search().strip()) > 0:
        search_term = input.table_search().strip().lower()
        searchable_columns = [
            "full_name",
            "owner",
            "description",
            "language",
            "license",
            "university",
            "affiliation_prediction_gpt_5_mini",
        ]
        mask = pd.Series([False] * len(data), index=data.index)
        for col in searchable_columns:
            if col in data.columns:
                mask |= data[col].astype(str).str.lower().str.contains(
                    search_term, na=False
                )
        data = data[mask]
    return data


@reactive.calc
def security_repositories_table_df():
    """
    One row per filtered repository: ``html_url`` plus scorecard columns from
    ``df_security`` (left join on ``html_url``).
    """
    base = filtered_df().drop(columns=_REPO_TABLE_DROP_COLS, errors="ignore")
    if "html_url" in base.columns:
        work = base[["html_url"]].copy()
    else:
        work = pd.DataFrame(index=base.index)
        work["html_url"] = pd.NA

    metric_pairs = [
        (d, s)
        for d, s in SECURITY_SCORECARD_METRICS
        if s in df_security.columns
    ]
    can_merge = (
        not df_security.empty
        and "html_url" in df_security.columns
        and bool(metric_pairs)
    )

    if can_merge:
        s_cols = ["html_url"] + [s for _, s in metric_pairs]
        sec = df_security[s_cols].drop_duplicates(subset=["html_url"], keep="first")
        out = work.merge(sec, on="html_url", how="left")
        out = out.rename(columns={s: d for d, s in metric_pairs})
    else:
        out = work.copy()

    for d, _ in SECURITY_SCORECARD_METRICS:
        if d not in out.columns:
            out[d] = pd.NA

    _total_col = "Total score"
    if _total_col in out.columns:
        out = out.sort_values(
            by=_total_col,
            key=lambda s: pd.to_numeric(s, errors="coerce"),
            ascending=False,
            na_position="last",
        )

    metric_displays = [d for d, _ in SECURITY_SCORECARD_METRICS]
    return out[["html_url"] + metric_displays]


@reactive.calc
def security_metric_averages_df():
    """
    One row per scorecard metric: mean of numeric values, excluding −1 and non-finite.
    """
    wide = security_repositories_table_df()
    rows = []
    for disp, _src in SECURITY_SCORECARD_METRICS:
        if disp not in wide.columns:
            rows.append((disp, float("nan")))
            continue
        s = pd.to_numeric(wide[disp], errors="coerce")
        s = s[s.notna() & (s != -1)]
        avg = float(s.mean()) if len(s) > 0 else float("nan")
        rows.append((disp, avg))
    out = pd.DataFrame(rows, columns=["Metric", "Average"])
    total_mask = out["Metric"].eq("Total score")
    main = out.loc[~total_mask].sort_values(
        "Average", ascending=False, na_position="last"
    )
    total_row = out.loc[total_mask]
    if total_row.empty:
        return main
    sep_row = pd.DataFrame([{"Metric": " ", "Average": float("nan")}])
    return pd.concat([main, sep_row, total_row], ignore_index=True)

