#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import polars as pl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import string
import matplotlib
import random
import json

acronyms = [
    "UCB", "UCD", "UCI", "UCLA", "UCM",
    "UCR", "UCSB", "UCSC", "UCSD", "UCSF"
]

matplotlib.rcParams['font.family'] = 'Lato'


def _sqlite_to_polars(conn: sqlite3.Connection, query: str) -> pl.DataFrame:
    """Execute a SQL query on a sqlite3 connection and return a Polars DataFrame."""
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    if not rows:
        return pl.DataFrame({col: [] for col in columns})
    return pl.DataFrame(rows, schema=columns, orient="row")


def db_to_df(db_path, output_filename, db_type='sqlite', db_params=None):
    """
    Exports all data from 'repositories' to a DataFrame, including org URL and email.

    :param db_path: Path to the SQLite database.
    :param output_filename: Base name for the CSV file (unused here).
    :param db_type: Type of database ('sqlite' only for now).
    :param db_params: Dictionary with connection parameters for PostgreSQL (unused).
    :return: A Polars DataFrame with repository info and organization contact details.
    """
    if db_type == 'sqlite':
        conn = sqlite3.connect(db_path)
    else:
        raise ValueError("Unsupported database type. Use 'sqlite'.")

    merged_df = _sqlite_to_polars(conn, "SELECT * FROM repositories")
    conn.close()

    # Clean up
    merged_df = merged_df.with_columns([
        pl.col("subscribers_count").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64),
        pl.col("release_downloads").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64),
    ]).drop("login", strict=False)

    return merged_df


def get_acronym_domain_map():
    acronym_domain = {}
    for acronym in acronyms:
        config_path = f"config/config_{acronym}.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            domain = config.get("UNIVERSITY_EMAIL_DOMAIN")
            if domain:
                acronym_domain[acronym] = domain
    return acronym_domain


def db_to_df_filtered(db_path, acronym, db_type='sqlite', db_params=None):
    """
    Exports all data from 'repositories' to a DataFrame, including org URL and email.

    :param db_path: Path to the SQLite database.
    :param acronym: University acronym used to filter by email/URL domain.
    :param db_type: Type of database ('sqlite' only for now).
    :param db_params: Dictionary with connection parameters for PostgreSQL (unused).
    :return: A Polars DataFrame with repository info and organization contact details.
    """
    if db_type == 'sqlite':
        conn = sqlite3.connect(db_path)
    else:
        raise ValueError("Unsupported database type. Use 'sqlite'.")

    acronym_domain = get_acronym_domain_map()

    repo_df = _sqlite_to_polars(conn, "SELECT * FROM repositories")
    org_df = _sqlite_to_polars(
        conn,
        "SELECT login, url AS org_url, email AS org_email FROM organizations"
    )
    conn.close()

    # Merge on owner == login
    merged_df = repo_df.join(org_df, left_on="owner", right_on="login", how="left")

    domain = acronym_domain[acronym]
    merged_df = merged_df.filter(
        pl.col("org_email").fill_null("").str.contains(domain, literal=True)
        | pl.col("org_url").fill_null("").str.contains(domain, literal=True)
    )

    # Clean up
    merged_df = merged_df.with_columns([
        pl.col("subscribers_count").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64),
        pl.col("release_downloads").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64),
    ]).drop("login", strict=False)

    return merged_df


def _keep_non_flag(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Keep rows where col is falsy (0/False) or null — drop truthy rows."""
    num = pl.col(col).cast(pl.Float64, strict=False)
    return df.filter((num == 0) | num.is_null())


def filter_data(data, threshold):
    """
    Filter data based on prediction threshold and repository characteristics.

    Filters the input data to include only rows where:
    - The 'affiliation_prediction_gpt_5_mini' column is greater than the threshold
    - Repository size is greater than 0
    - Repository is not archived
    - Repository is not a fork
    - Repository has at least one star

    Parameters
    ----------
    data : polars.DataFrame
        The input dataset containing repository data.
    threshold : float
        Minimum prediction value to include.

    Returns
    -------
    polars.DataFrame
        The filtered dataset.
    """
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)

    # Cast prediction column to numeric
    data = data.with_columns(
        pl.col("affiliation_prediction_gpt_5_mini").cast(pl.Float64, strict=False)
    )

    # Apply threshold filter
    filtered = data.filter(pl.col("affiliation_prediction_gpt_5_mini") > threshold)

    # Filter out repositories with size=0 (keep size=NULL)
    if "size" in filtered.columns:
        filtered = filtered.with_columns(
            pl.col("size").cast(pl.Float64, strict=False)
        ).filter(
            (pl.col("size") > 0) | pl.col("size").is_null()
        )

    # Filter out archived repositories (keep only non-archived: 0, False, or null)
    if "archived" in filtered.columns:
        filtered = _keep_non_flag(filtered, "archived")

    # Filter out fork repositories (keep only non-forks: 0, False, or null)
    if "fork" in filtered.columns:
        filtered = _keep_non_flag(filtered, "fork")

    # Filter out template repositories (keep only non-templates: 0, False, or null)
    if "is_template" in filtered.columns:
        filtered = _keep_non_flag(filtered, "is_template")

    return filtered


def build_shared_color_map(all_data_dict, column, threshold=0.02):
    """
    Build a shared color map for a specified column across multiple DataFrames.

    Parameters:
    ----------
    all_data_dict : dict of {str: polars.DataFrame}
        A dictionary mapping university acronyms to their corresponding DataFrames.
    column : str
        The column name for which the color map is to be generated.
    threshold : float, optional
        The minimum proportion a label must have to be assigned its own color. Default 0.02.

    Returns:
    -------
    dict
        A dictionary mapping labels to color values.
    """
    LANGUAGE_LABEL_MAP = {
        "Jupyter Notebook": "Jupyter",
    }

    # Concatenate the column across all DataFrames, applying label normalization
    parts = []
    for df in all_data_dict.values():
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        series = df[column].replace(
            list(LANGUAGE_LABEL_MAP.keys()),
            list(LANGUAGE_LABEL_MAP.values()),
        )
        parts.append(series)

    combined = pl.concat(parts)
    total = combined.len()

    value_counts = (
        pl.DataFrame({column: combined})
        .filter(pl.col(column).is_not_null())
        .group_by(column)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    # Keep only labels that exceed the threshold globally
    major_labels = (
        value_counts
        .filter(pl.col("count") / total >= threshold)
        [column]
        .drop_nulls()
        .to_list()
    )

    unique_labels = sorted(major_labels)

    if threshold > 0:
        unique_labels.append("Other")
    if column == "license":
        unique_labels.append("None")

    # Generate color map
    cmap = plt.colormaps['tab20'].resampled(len(unique_labels))
    return dict(zip(unique_labels, [cmap(i) for i in range(len(unique_labels))]))
