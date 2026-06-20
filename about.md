## **What it is**

The Open Source Repository Browser is an interactive dashboard for exploring open source software on GitHub that is connected to universities and research organizations. You can slice the collection by institution, technology, and impact metrics, drill into individual repositories, and compare patterns across the ecosystem.

The goal is simple: make it easier to **see** how open source from academic and affiliated communities is distributed, how projects are maintained, and where sustainability and security practices show up, without digging through GitHub one repo at a time.

---

## **Where the data comes from**

### Discovery on GitHub

The underlying dataset is built from **GitHub**. Repositories are collected by **scraping and querying** GitHub for users, organizations, and repositories that match a curated set of institutions and naming patterns. Using **regular expressions**, **university names and acronyms** are matched against GitHub entities (users, organizations, and repositories) to find candidate projects tied to those institutions.

#### Current Data

The **current dataset** for this dashboard is scoped to the **universities and institutions represented in CURIOSS**, a community for individuals who work in university and research institution OSPOs (see the **[current CURIOSS members list](https://curioss.org/about/members/)**).

### Affiliation and project type (AI-assisted)

Not every match is a clean institutional link. **Large language models** from **OpenAI**, specifically **GPT‑5-mini** were used to estimate how strongly a repository is affiliated with a given university given the repository **README** and **description**; **Contributor** and profile-related signals where available. Additionally, repositories project types were predicted using the same methodology.

You’ll see these as **affiliation probabilities** and **project type** labels in the app (for example, development tooling vs. other categories). They are **predictions**, not ground truth, use them as guidance alongside the raw metadata.

For more information on the data collection and classification, see **[Recipe for Discovery: A Pipeline for Institutional Open Source Activity](https://arxiv.org/abs/2506.18359)** (arXiv preprint).

### What is filtered out (for a clearer, faster app)

To keep the dataset focused and the application responsive, the build **excludes** several kinds of repositories: **Archived** repositories, **forks**, **repository templates**, and, for now, repositories with **zero stars** (so the browser concentrates on projects that have attracted at least minimal attention).

### Security scores (OpenSSF Scorecard)

**[OpenSSF Scorecard](https://scorecard.dev/)** checks are run against projects to bring in security-related metrics (for example branch protection, CI, signed releases, and an overall score). **Coverage is still incomplete**, only a **subset** of repositories currently has scorecard results. **Filling this in for the full dataset is ongoing work**, and the Security views will become more complete as that pipeline finishes.

---

## **Data Collection and Usage Policies**

### Data Collection Methodology
- All data was collected exclusively using the GitHub API.
- No scraping was performed outside of GitHub’s provided interfaces.
- No data was collected from sources other than GitHub.

### Compliance with GitHub Terms of Service and API Policies
- The project operates in compliance with **[GitHub’s Terms of Service and API usage policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies#7-information-usage-restrictions)**. 

### Nature of UC Affiliation Claims
- The database does not claim authoritative or official affiliation between repositories/contributors and the UC system. Associations are inferred using:
    - Keyword matching
    - Repository metadata (e.g., descriptions, READMEs)
    - Automated classification techniques, including LLM-assisted methods

These associations are best-effort signals, not ground truth.

### Data Scope Limitations
- Only public GitHub data is included.
- No private repositories, private profiles, or off-platform information are used.

### Opt-Out and Contact Mechanism
- If you have questions or concerns about the data or you want your project to be removed from this dashboard please contact **jgomez91@ucsc.edu**. 

---

## **Using the dashboard**

### Main tabs: what you’ll see

The dashboard is organized into two main tabs, **Repositories** and **Organizations**, each with its own sidebar filters and inner sub-tabs.

---

## **Repositories tab**

### Sidebar: filters

All views inside the Repositories tab respond to the **same filter set**. On the left you’ll find:

- **Prediction threshold**: Keeps repositories whose **estimated affiliation probability** falls in the range you choose (for example, high-confidence matches only). The default value for this metric is 0.8.
- **University**: One or more institutions to include.
- **Project type**: Filter by the **predicted project type** from the model.
- **License**: Filter by declared **license**.
- **Language**: Filter by primary **programming language**.
- **# Stars / # Forks / # Release downloads**: **Sliders** to restrict repositories by those numeric ranges.

Use **Reset all filters** to clear selections and ranges and start over.

### Inner tabs

#### Overview

High-level **summary numbers**: how many repositories and contributors are in view, what share has a license, and average **bus factor** (a simple resilience signal). Below that:

A **table** of repository counts **per university**; charts for **community files** (README, contributing guide, security policy, etc.); and **project type**, **language**, and **license** distributions, including breakdowns **by project type** where it helps compare segments.

This tab is a good first stop for “what’s in the box?” after you set filters.

#### Browse

A **searchable table** of repositories matching your filters. **Click a row** to open a detail panel:

- **Overview**: name, university, license, language, type, description, and link to GitHub
- **Impact**: stars, forks, release downloads, issues, and contributors
- **Health**: quick checks for description, README, contributing guide, code of conduct, security policy, and issue/PR templates
- **Security**: OpenSSF Scorecard metrics when available for that repo.

Alongside that, tabs show the **README**, **Contributing** guide, and **Security policy** as rendered markdown when present. You can **download** the filtered repository list as CSV from this area.

#### Impact

**Totals** across the filtered set for stars, forks, release downloads, and contributors. An **Impact indicators per university** table ranks institutions by those measures. **Distribution charts** show how stars, forks, downloads, and contributors spread across buckets.

#### Sustainability

Focuses on **maintainability-style** signals: **average contributors** and **average bus factor** per university in a leaderboard table, plus summary value boxes. Charts include **community feature** presence by project type, a **heatmap** of features for **development (DEV)**-typed repos across **star** ranges, and distributions of **bus factor** and **contributor count** buckets.

#### Security

Two complementary views:

1. A **wide table** of repositories with **OpenSSF Scorecard** columns (scores align with **[scorecard.dev](https://scorecard.dev/)**).
2. An **average score per metric** visualization so you can see which security checks tend to score higher or lower across the filtered repositories.

Remember: **many cells may be empty** until scorecard coverage catches up with the full repository list.

---

## **Organizations tab**

Coming soon. This tab will surface data about the GitHub **organizations** affiliated with CURIOSS member institutions, including activity, membership, and repository breakdowns at the organization level.

---

## **A quick note on limitations**

- **Affiliation and type** come from models and heuristics, so they can be wrong or outdated. 
- **GitHub** metadata changes, so some numbers might be outdated. Periodic refresh from GitHub is planned.
- **Scorecard** results are **partial** today but are intended to grow toward full coverage.

---

**Contact**

Questions about the data, requests to correct information, or to have a repository removed from this dashboard: [jgomez91@ucsc.edu](mailto:jgomez91@ucsc.edu).

For other feedback or bug reports, please [open an issue on GitHub](https://github.com/juanis2112/repoexplorer).
