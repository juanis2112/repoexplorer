#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from shiny.express import input, ui, render
from shiny import reactive
from shiny import session as shiny_session
from shiny import ui as sui
from faicons import icon_svg
import matplotlib.pyplot as plt
import io
import json
import logging
import querychat as qc
from repoexplorer.analysis.type_distribution import plot_type_distribution
from repoexplorer.analysis.language_distribution_by_type import plot_language_distribution_by_type
from repoexplorer.analysis.language_distribution import plot_language_distribution
from repoexplorer.analysis.license_distribution_by_type import plot_license_distribution_by_type
from repoexplorer.analysis.license_distribution import plot_license_distribution
from repoexplorer.analysis.feature_counts_per_type import plot_feature_counts_per_type
from repoexplorer.analysis.feature_counts import plot_feature_counts
from repoexplorer.analysis.university_distribution import plot_university_distribution
from repoexplorer.analysis.feature_heatmap_per_stars import plot_feature_heatmap_by_star_bucket
from repoexplorer.analysis.commit_history import plot_commit_history
from dotenv import load_dotenv

load_dotenv()


if "OPENAI_MODEL" not in os.environ:
    os.environ["OPENAI_MODEL"] = "gpt-5-mini"

# Data/parquet/{acronym}/repositories.parquet (case-insensitive acronym match)
# Optional fast path: Data/parquet/repositories_combined.parquet (single pre-merged file)
PARQUET_BASE = "Data/parquet"
COMBINED_PARQUET = os.path.join(PARQUET_BASE, "repositories_combined_clean.parquet")
SECURITY_PARQUET = os.path.join(PARQUET_BASE, "security_combined_clean.parquet")
ORGANIZATIONS_PARQUET = os.path.join(PARQUET_BASE, "organizations_combined_clean.parquet")
CONTRIBUTORS_PARQUET = os.path.join(PARQUET_BASE, "contributors_combined_clean.parquet")
# Commits table (use existing combined, cleaned file)
COMMITS_PARQUET = os.path.join(PARQUET_BASE, "commits_combined_clean.parquet")

# Columns to load (fewer columns = faster load). "university" is added from config.
COLUMNS_TO_LOAD = [
    "id", "full_name", "owner", "license", "language", "html_url", "description", "fork", "created_at",
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

# Remove stray "True" that can appear from querychat/Shiny return values (This is hacky but leaving it for now)
ui.tags.script("""
(function() {
  function removeTrueNodes(node) {
    if (!node) return;
    if (node.nodeType === Node.TEXT_NODE && node.textContent.trim() === "True") {
      node.parentNode.removeChild(node);
      return;
    }
    if (node.nodeType === Node.ELEMENT_NODE && node.childNodes.length === 1 &&
        node.childNodes[0].nodeType === Node.TEXT_NODE && node.childNodes[0].textContent.trim() === "True") {
      node.parentNode.removeChild(node);
      return;
    }
    for (var i = node.childNodes.length - 1; i >= 0; i--) {
      removeTrueNodes(node.childNodes[i]);
    }
  }
  function run() {
    removeTrueNodes(document.body);
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


def _load_one_acronym(acronym: str, parquet_dir: str):
    """
    Load a repository parquet file and add the university name.

    Parameters
    ----------
    acronym : str
        University acronym (for example, ``"UCB"``).
    parquet_dir : str
        Subdirectory under ``PARQUET_BASE`` that contains ``repositories.parquet``.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with the requested columns and a ``"university"`` column,
        or ``None`` if the parquet file is missing or cannot be read.
    """
    repo_path = os.path.join(PARQUET_BASE, parquet_dir, "repositories.parquet")
    if not os.path.isfile(repo_path):
        return None
    try:
        # Read only needed columns for faster I/O
        df = pd.read_parquet(repo_path, columns=[c for c in COLUMNS_TO_LOAD if c != "university"])
    except Exception:
        df = pd.read_parquet(repo_path)
        df = df[[c for c in COLUMNS_TO_LOAD if c in df.columns and c != "university"]]
    config_file = f"config/config_{acronym.replace(' ', '_')}.json"
    if os.path.isfile(config_file):
        try:
            with open(config_file, encoding="utf-8") as f:
                df["university"] = json.load(f).get("UNIVERSITY_NAME", acronym)
        except Exception:
            df["university"] = acronym
    else:
        df["university"] = acronym
    return df


# Fast path: single pre-merged parquet 
if os.path.isfile(COMBINED_PARQUET):
    try:
        cload = COLUMNS_TO_LOAD + ["university"]
        df = pd.read_parquet(COMBINED_PARQUET, columns=cload)
        if "university" not in df.columns:
            df["university"] = "Unknown"
    except Exception:
        logging.exception("Failed to load combined parquet %s", COMBINED_PARQUET)
        df = pd.DataFrame()
else:
    # Build case-insensitive map: parquet subdir name (lower) -> actual subdir name
    _parquet_dirs = {}
    if os.path.isdir(PARQUET_BASE):
        for name in os.listdir(PARQUET_BASE):
            if os.path.isdir(os.path.join(PARQUET_BASE, name)):
                _parquet_dirs[name.lower()] = name

    df_list = []
    with ThreadPoolExecutor(max_workers=min(8, len(ACRONYMS) or 1)) as executor:
        submitted = []
        for acronym in ACRONYMS:
            key = acronym.replace(" ", "_").lower()
            parquet_dir = _parquet_dirs.get(key)
            if not parquet_dir:
                continue
            submitted.append((executor.submit(_load_one_acronym, acronym, parquet_dir), acronym))
        for future, acronym in submitted:
            try:
                result = future.result()
                if result is not None:
                    df_list.append(result)
                else:
                    logging.error("Failed to load parquet for acronym %s", acronym)
            except Exception:
                logging.exception("Failed to load parquet for acronym %s", acronym)
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# Load security_combined table (OpenSSF scores)
df_security = pd.DataFrame()
if os.path.isfile(SECURITY_PARQUET):
    try:
        df_security = pd.read_parquet(SECURITY_PARQUET)
    except Exception:
        logging.exception("Failed to load security parquet %s", SECURITY_PARQUET)
        df_security = pd.DataFrame()

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
df_commits = pd.DataFrame()
if os.path.isfile(COMMITS_PARQUET):
    try:
        df_commits = pd.read_parquet(COMMITS_PARQUET)
    except Exception:
        logging.exception("Failed to load commits parquet %s", COMMITS_PARQUET)
        df_commits = pd.DataFrame()


# =============================================== App UI ==========================================

ui.page_opts(title="University Repositories", fillable=True)

# ======================================== Sidebar  ===============================================

# ------------------------------------ Manual Filters ----------------------------------------------

licenses = df["license"].unique().tolist() if "license" in df.columns else []
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
_greeting_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greeting.md")
with open(_greeting_path, encoding="utf-8") as _f:
    _greeting_md = _f.read()

querychat_config = qc.init(
    data_source=_df_08,
    table_name="Repositories",
    greeting=_greeting_md,
)

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
            
            

       with ui.nav_panel("Chat Bot"):
            qc.ui("chat")  # may return True; do not use as last expression
   
   # Reset button at the bottom of the sidebar
   ui.br()
   ui.br()
   ui.input_action_button("reset_filters", "Reset All Filters", class_="btn-danger")
   ui.HTML("")  # prevent any prior return value (e.g. True) from rendering

# Assign chat server in a function so its return value is not rendered as "True" in the main panel
chat = None


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
    try:
        sess = shiny_session.get_current_session()
        if sess is not None:
            # Send message to chat input so the bot receives "reset all filters" and can clear filters
            sess.send_input_message("chat-message", {"value": "reset all filters"})
    except Exception:
        pass

# ======================================== Main panel  ===============================================

#------------------------------------ Overview ----------------------------------------------


# Icons for Overview value boxes
ICONS = {
    "repos": icon_svg("code-branch"),
    "contributors": icon_svg("users"),
    "active": icon_svg("clock"),
    "openssf": icon_svg("shield-halved"),
    "busfactor": icon_svg("bus"),
    "license": icon_svg("id-card"),
}

with ui.navset_pill(id="tab"):  
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
                            lic = data["license"].astype(str).str.strip()
                            with_license = ((lic != "") & (lic.str.lower() != "none")).sum()
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
        with ui.layout_columns(col_widths=(7, 5)):
            
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
            
            with ui.card():
                @render.plot
                def plot_files_combined():
                    return _make_feature_counts_combined_fig(
                        filtered_df(),
                        FEATURES,
                        acronym="",
                        figsize=(8, 6),
                        label_size=10,
                        title_size=12,
                        textprops=8,
                    )

                @render.download(
                    filename=lambda: f"community_files_presence_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                def download_plot_files_combined():
                    fig = _make_feature_counts_combined_fig(
                        filtered_df(),
                        FEATURES,
                        acronym="",
                        figsize=(8, 6),
                        label_size=11,
                        title_size=12,
                        textprops=10,
                    )
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    plt.close(fig)
                    yield buf.read()

        with ui.layout_columns(col_widths=(4, 4, 4)):  
            with ui.card():
                @render.plot
                def plot_type():
                    fig, ax = plt.subplots(figsize=(8,6))
                    plot_type_distribution(
                        filtered_df(), 
                        acronym="University", 
                        ax=ax, 
                        label_size=10, 
                        title_size=12,
                        textprops=8)
                    return fig

                @render.download(filename=lambda: f"project_distribution_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
                def download_plot_type_file():
                    fig, ax = plt.subplots(figsize=(8,6))
                    plot_type_distribution(
                        filtered_df(), 
                        acronym="", 
                        ax=ax, 
                        label_size=11, 
                        title_size=12,
                        textprops=10)
                    # Save figure to bytes
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    yield buf.read()
            with ui.card():
                @render.plot
                def plot_language_combined():
                    return _make_language_combined_fig(
                        filtered_df(),
                        acronym="",
                        figsize=(8, 6),
                        label_size=10,
                        title_size=12,
                        props=9,
                        other_thres=0.1,
                    )

                @render.download(
                    filename=lambda: f"language_distribution_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                def download_plot_language_combined():
                    fig = _make_language_combined_fig(
                        filtered_df(),
                        acronym="",
                        figsize=(8, 6),
                        label_size=11,
                        title_size=12,
                        props=11,
                        other_thres=0.1,
                    )
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    plt.close(fig)
                    yield buf.read()

            with ui.card():
                @render.plot
                def plot_license_combined():
                    return _make_license_combined_fig(
                        filtered_df(),
                        acronym="",
                        figsize=(8, 6),
                        label_size=10,
                        title_size=12,
                        textprops=7,
                        other_thres=0.009,
                    )

                @render.download(
                    filename=lambda: f"license_distribution_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                def download_plot_license_combined():
                    fig = _make_license_combined_fig(
                        filtered_df(),
                        acronym="",
                        figsize=(8, 6),
                        label_size=11,
                        title_size=12,
                        textprops=9,
                        other_thres=0.009,
                    )
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    plt.close(fig)
                    yield buf.read()

#------------------------------------ Repositories ----------------------------------------------
    
    
    with ui.nav_panel("Repositories"):
        with ui.card(class_="repo-data-card"):
            ui.input_text("table_search", "Search", placeholder="Search repositories...", width="100%")
            @render.data_frame  
            def display_df():
                data = filtered_df().drop(columns=["readme", "contributing", "contributors", 'code_of_conduct_file', 'contributing', 'security_policy', 'issue_templates', 'pull_request_template'], errors="ignore")
                
                # Apply search filter if search term is provided
                if input.table_search() and len(input.table_search().strip()) > 0:
                    search_term = input.table_search().strip().lower()
                    # Search across multiple text columns
                    searchable_columns = ['full_name', 'owner', 'description', 'language', 'license', 'university', 'affiliation_prediction_gpt_5_mini']
                    mask = pd.Series([False] * len(data), index=data.index)
                    
                    for col in searchable_columns:
                        if col in data.columns:
                            # Convert to string and search (case-insensitive)
                            mask |= data[col].astype(str).str.lower().str.contains(search_term, na=False)
                    
                    data = data[mask]
                
                return render.DataGrid(
                    data,
                    height="500px",
                    selection_mode="row",
                )
        with ui.card():
            @render.ui
            def show_clicked():
                selected_rows = display_df.cell_selection()["rows"]

                if not selected_rows:
                    return ""

                # Row index refers to the displayed (filtered) table, not the full df
                row_idx = selected_rows[0]
                data = filtered_df()
                selected = data.iloc[row_idx]
            
                # Safely get impact-related metrics, falling back to 'N/A' if missing
                stars = selected.get("stargazers_count", "N/A")
                downloads = selected.get("release_downloads", "N/A")
                forks_count = selected.get("forks_count", "N/A")
                num_issues = selected.get("open_issues_count", "N/A")
                num_contributors = selected.get("contributor_count", "N/A")


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
                                    selected["full_name"],
                                ),
                                ui.p(
                                    ui.tags.span("University: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                    selected.get("university", "Unknown"),
                                ),
                                ui.p(
                                    ui.tags.span("License: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                    selected["license"],
                                ),
                                ui.p(
                                    ui.tags.span("Language: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                    selected["language"],
                                ),
                                ui.p(
                                    ui.tags.span("Project Type: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                    selected["type_prediction_gpt_5_mini"],
                                ),
                                ui.p(
                                    ui.tags.span("Description: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                    selected["description"],
                                ),
                                ui.p(
                                    ui.tags.span("URL: ", style="color: var(--bs-primary, #0d6efd); font-weight: bold;"),
                                    ui.tags.a(selected["html_url"], href=selected["html_url"], target="_blank"),
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
                                        ui.tags.td(str(int(stars)), style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Number of downloads",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td(str(int(downloads)), style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Number of forks",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td(str(int(forks_count)), style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Number of issues",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td(str(int(num_issues)), style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Number of contributors",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td(str(int(num_contributors)), style="text-align: center;"),
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
                                        ui.tags.td("✅" if selected.get("description") else "✗", style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "README",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td("✅" if selected.get("readme") else "✗", style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Contributing guide",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td("✅" if selected.get("contributing") else "✗", style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Code of conduct",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td("✅" if selected.get("code_of_conduct_file") else "✗", style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Security policy",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td("✅" if selected.get("security_policy") else "✗", style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "Issue templates",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td("✅" if selected.get("issue_templates") else "✗", style="text-align: center;"),
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td(
                                            "PR template",
                                            style=(
                                                "padding-right: 6px; "
                                                "color: var(--bs-primary, #0d6efd); font-weight: bold;"
                                            ),
                                        ),
                                        ui.tags.td("✅" if selected.get("pull_request_template") else "✗", style="text-align: center;"),
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
                                                ui.tags.td(str(sec_row.get(col, "—")), style="text-align: center;"),
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
                                ui.markdown(selected.get("readme", "") or "")
                                if selected.get("readme")
                                else ui.p("No README available", class_="text-muted"),
                            ),
                            sui.nav_panel(
                                "Contributing",
                                ui.markdown(selected.get("contributing", "") or "")
                                if selected.get("contributing")
                                else ui.p("No contributing guide available", class_="text-muted"),
                            ),
                            sui.nav_panel(
                                "Security Policy",
                                ui.markdown(selected.get("security_policy", "") or "")
                                if selected.get("security_policy")
                                else ui.p("No security policy available", class_="text-muted"),
                            ),
                            id="repo_detail_bottom",
                        ),
                        style="padding-left: 16px;",
                    ),
                    col_widths=(4, 6),
                )
        
    with ui.nav_panel("Insights"):
        # First row: feature counts per type & license distribution
        with ui.layout_columns():
            with ui.card():
                @render.plot
                def plot_files():
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_feature_counts_per_type(
                        filtered_df(),
                        FEATURES,
                        acronym="",
                        ax=ax,
                        label_size=8,
                        title_size=8,
                        textprops=8,
                    )
                    return fig

            with ui.card():
                @render.plot
                def plot_license():
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_license_distribution_by_type(
                        filtered_df(),
                        acronym="",
                        ax=ax,
                        label_size=8,
                        title_size=8,
                        textprops=8,
                        other_thres=0.009,
                    )
                    return fig

        # Second row: language distribution & feature heatmap
        with ui.layout_columns():
            with ui.card():
                @render.plot
                def plot_language():
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_language_distribution_by_type(
                        filtered_df(),
                        acronym="",
                        ax=ax,
                        label_size=8,
                        title_size=8,
                        props=7,
                        other_thres=0.02,
                    )
                    return fig

            with ui.card():
                @render.plot
                def plot_heatmap():
                    dev_df = filtered_df()[filtered_df()["type_prediction_gpt_5_mini"] == "DEV"]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_feature_heatmap_by_star_bucket(
                        dev_df,
                        FEATURES,
                        star_col="stargazers_count",
                        ax=ax,
                        label_size=8,
                        title_size=8,
                        annotations_size=8,
                    )
                    return fig

# ------------------------------------ Filtered DataFrame ----------------------------------------------

@reactive.calc  
def filtered_df():
    result = df.copy()
    
    # Apply manual filters
    if input.university(): 
        result = result.loc[result.university.isin(input.university())]
    if input.type(): 
        result = result.loc[result.type_prediction_gpt_5_mini.isin(input.type())]
    if input.license(): 
        result = result.loc[result.license.isin(input.license())]
    if input.language(): 
        result = result.loc[result.language.isin(input.language())]
    if input.slider_stars():
        min_val, max_val = input.slider_stars()
        result["stargazers_count"] = pd.to_numeric(result["stargazers_count"], errors="coerce")
        result = result[
            (result["stargazers_count"] >= min_val) &
            (result["stargazers_count"] <= max_val)
        ]
    if input.slider_forks():
        min_val, max_val = input.slider_forks()
        result["forks_count"] = pd.to_numeric(result["forks_count"], errors="coerce")
        result = result[
            (result["forks_count"] >= min_val) &
            (result["forks_count"] <= max_val)
        ]
    
    if input.slider_downloads():
        min_val, max_val = input.slider_downloads()
        result["release_downloads"] = pd.to_numeric(result["release_downloads"], errors="coerce")
        result = result[
            (result["release_downloads"] >= min_val) &
            (result["release_downloads"] <= max_val)
        ]

    if input.slider_threshold():
        min_val, max_val = input.slider_threshold()
        result["affiliation_prediction_gpt_5_mini"] = pd.to_numeric(result["affiliation_prediction_gpt_5_mini"], errors="coerce")
        result = result[
            (result["affiliation_prediction_gpt_5_mini"] >= min_val) &
            (result["affiliation_prediction_gpt_5_mini"] <= max_val)
        ]

    # Apply chat filters - combine with manual filters (only when chat is active and has results)
    if chat is not None:
        try:
            chat_df = chat.df()
            if chat_df is not None and len(chat_df) > 0:
                # Get the IDs from the chat results
                if 'id' in chat_df.columns:
                    chat_ids = set(chat_df['id'].values)
                    result = result[result['id'].isin(chat_ids)]
                elif 'id' in result.columns:
                    chat_indices = set(chat_df.index)
                    result = result[result.index.isin(chat_indices)]
                else:
                    chat_indices = set(chat_df.index)
                    result = result[result.index.isin(chat_indices)]
        except Exception:
            # If chat reset or .df() unavailable, use only manual filters
            pass

    return result

