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

This tab surfaces data about the GitHub **organizations** affiliated with CURIOSS member institutions.

### Sidebar: filters

All views inside the Organizations tab respond to the same filter set. On the left you'll find:

**Prediction threshold**: Keeps organizations whose estimated affiliation probability falls in the range you choose. The default range is 0.5 to 1.0, reflecting that organization-level affiliation signals tend to be noisier than repository-level ones.

**University**: One or more institutions to include.

Use **Reset Org Filters** to clear all selections and start over.

### Inner tabs

#### Overview

High-level summary numbers: total organizations in view, along with the share of organizations that have a description, a URL, and an email address on record. Below that, charts show the distribution of organizations per university, the number of organizations created per year, and profile completeness across key fields (description, location, website, email, and company).

#### Browse

A searchable table of organizations matching your filters. Fields shown include login, GitHub URL, name, university, description, company, email, URL, location, creation date, and affiliation score.

### Affiliation scoring for organizations

Each organization receives an **affiliation score** between 0 and 1, estimated using a rule-based system that checks the organization's login, name, description, email, URL, location, and company field against each institution's known domain, website, acronym, and name. The tiers are as follows.

**Score 1.0** is assigned when the organization's email contains the university domain, when the URL or company field contains the university domain or website, or when the description contains the university domain, website URL, or acronym as a whole word.

**Score 0.90** is assigned when the university acronym (three or more characters) appears in the login.

**Score 0.88** is assigned when the acronym appears in the name or company field.

**Score 0.85** is assigned when at least 50% of the significant words in the university name (words longer than three characters) appear in the login.

**Score 0.82** applies the same word-overlap test against the name field.

**Score 0.80** applies the same word-overlap test against the company field.

**Score 0.60** is assigned when the first distinctive word of the university name (longer than four characters) appears anywhere across all fields.

**Score 0.55** is assigned when the university city appears in the location field alongside at least one other weak signal (domain, acronym, or first distinctive word present anywhere).

**Score 0.20** is the base score for organizations that were scraped for a given institution but show no textual confirmation of affiliation.

Checks are applied in the order above and return at the first match, so higher tiers always win. These are estimates, not ground truth, and should be used as guidance alongside the raw metadata.

---

## **A quick note on limitations**

- **Affiliation and type** come from models and heuristics, so they can be wrong or outdated. 
- **GitHub** metadata changes, so some numbers might be outdated. Periodic refresh from GitHub is planned.
- **Scorecard** results are **partial** today but are intended to grow toward full coverage.

---

**Contact**

Questions about the data, requests to correct information, or to have a repository removed from this dashboard: [jgomez91@ucsc.edu](mailto:jgomez91@ucsc.edu).

For other feedback or bug reports, please [open an issue on GitHub](https://github.com/juanis2112/repoexplorer).
