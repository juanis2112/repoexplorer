# Repoexplorer
Interactive Shiny dashboard for exploring university open source repositories, contributors, and activity.

## Simple Shiny app setup

This Shiny app lets you explore repositories and use an OpenAI-powered chat bot over the data.

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd repoexplorer  
```

### 2. Install dependencies 

From the project root:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes `shiny`, `pyarrow`, and `querychat` for the Shiny UI, Parquet data loading, and the chat bot.

### 3. Set up your OpenAI token (for the chat bot)

Set your OpenAI API key as an environment variable (or in a `.env` file):

```bash
export OPENAI_API_KEY=sk-...
```

Without this key, the app will still load, but the chat bot will be disabled.

### 4. Download the data

1. Download the Parquet data from [this Google Drive folder](https://drive.google.com/drive/folders/1I7mRHanT7dR2OZgHOL2Se-MSWKsewuwY?usp=sharing).
2. Place the downloaded files inside the `Data/parquet` folder (create it if it does not exist).

Your layout should look roughly like:

```bash
repoexplorer/
├── app.py
└── Data/
    └── parquet/
        └── ...
```

### 5. Run the app

From the project root, run:

```bash
shiny run --reload --launch-browser app.py
```

This will start the app, automatically open a browser window, and reload on code changes. That’s it.
