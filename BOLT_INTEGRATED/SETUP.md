# BOLT — Local Setup & Hosting Guide

**BOLT (Bayesian Optimised Load Trimmer)** is an AI-powered energy management platform that
forecasts electricity demand, shaves peak loads via battery dispatch, computes TNB bills, and
recommends optimal solar and battery sizing with 25-year financial returns.

This guide walks you through running the full system on your own machine — no cloud account,
no API key, and no special hardware required.

---

## Table of Contents

1. [What you need before you start](#1-what-you-need-before-you-start)
2. [Get the code](#2-get-the-code)
3. [Create an isolated Python environment](#3-create-an-isolated-python-environment)
4. [Install dependencies](#4-install-dependencies)
5. [Launch the app](#5-launch-the-app)
6. [First-run workflow](#6-first-run-workflow)
7. [File and folder structure](#7-file-and-folder-structure)
8. [Keeping the app running (longer sessions)](#8-keeping-the-app-running-longer-sessions)
9. [Troubleshooting](#9-troubleshooting)
10. [Updating to the latest version](#10-updating-to-the-latest-version)

---

## 1. What you need before you start

Install the following tools **in this order** before proceeding.

### Python 3.11 or later

BOLT requires Python 3.11 or newer (3.13 is recommended).

**Check if you already have it:**

```
python --version
```

If the output says `Python 3.11.x` or higher, you are good.
If it says `Python 3.9.x` or lower, or if the command is not found, install Python from:

> https://www.python.org/downloads/

During installation on Windows, tick **"Add Python to PATH"** before clicking Install.

---

### Git

Git is needed to download the code from GitHub.

**Check if you already have it:**

```
git --version
```

If not installed, download it from:

> https://git-scm.com/downloads

---

### pip (Python package manager)

pip is included with Python 3.11+. Verify it is working:

```
pip --version
```

If that fails, try:

```
python -m pip --version
```

---

## 2. Get the code

Open a terminal (Command Prompt, PowerShell, or any terminal on macOS/Linux) and run:

```
git clone https://github.com/JohnDoeSKipper/BOLT-bolt-integrated-ui.git
```

This downloads the entire repository into a new folder called `BOLT-bolt-integrated-ui`
in whichever directory your terminal is currently in.

Navigate into it:

```
cd BOLT-bolt-integrated-ui
```

---

## 3. Create an isolated Python environment

A virtual environment keeps BOLT's dependencies separate from everything else on your machine.
This prevents version conflicts and makes it easy to delete the environment later if needed.

**Create the environment** (run this once):

```
python -m venv .venv
```

This creates a hidden folder called `.venv` inside the repository.

**Activate the environment:**

*Windows (Command Prompt):*
```
.venv\Scripts\activate.bat
```

*Windows (PowerShell):*
```
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run this first and then retry:
> ```
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

*macOS / Linux:*
```
source .venv/bin/activate
```

You will know the environment is active when your terminal prompt starts with `(.venv)`.

**Important:** You must activate the environment every time you open a new terminal to run BOLT.
The `.venv` folder never changes — you only create it once.

---

## 4. Install dependencies

With the virtual environment active, navigate into the application folder and install all
required packages in one command:

```
cd BOLT_INTEGRATED
pip install -r requirements.txt
```

This downloads and installs approximately 15 packages including LightGBM (the forecasting
engine), Streamlit (the web interface), Plotly (charts), and supporting data science libraries.

Installation takes 1–3 minutes depending on your internet speed. You only need to do this once
(or again if `requirements.txt` is updated).

**Verify the install worked:**

```
python -c "import streamlit, lightgbm, plotly; print('All good.')"
```

You should see `All good.` with no errors.

---

## 5. Launch the app

Make sure you are still inside the `BOLT_INTEGRATED` folder (where `app.py` lives) and that
your virtual environment is active, then run:

```
streamlit run app.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Open your browser and go to:** `http://localhost:8501`

The app loads in 5–10 seconds. Leave the terminal open — closing it stops the app.

---

## 6. First-run workflow

The seven tabs must be used in a specific order. The system is designed so that each tab feeds
data into the next one. Follow this sequence on your first run.

---

### Step 1 — Site Setup tab

Configure your site before loading any data.

- Set the latitude and longitude of your site (used for weather data and solar irradiance).
  Default is Kuala Lumpur. If you are in Penang, for example, change to 5.4141, 100.3288.
- Enter your battery capacity (kWh), C-rate, and initial State of Charge.
- Set EV charger count, kW rating per charger, and the permitted charging window
  (e.g. 18:00 to 08:00 overnight).
- Set the HVAC protected hours (when HVAC cannot be cut, e.g. 09:00 to 17:00 business hours).
- Click **Apply changes** to save. These settings persist across browser refreshes.

---

### Step 2 — Data Upload tab

Upload your TNB meter export file. Accepted formats:

- Excel (`.xlsx`, `.xls`) — all standard TNB export formats (SOL, E, SUN, MI2)
- CSV (`.csv`) — any comma-separated file with a timestamp column and kW/kVAR columns

The system auto-detects column names and resamples to 30-minute intervals. After upload:

- A load profile chart appears.
- The system auto-detects your TNB tariff (A, B, C1, C2, D, E1, or E2).
- Data quality issues (gaps, spikes, zero runs) are flagged automatically.
- Your data is saved to disk, so you do not need to re-upload after a browser refresh.

**Minimum recommended dataset:** 60 days of half-hourly data (2,880 rows).
The forecaster can train on less but accuracy improves significantly with more history.

---

### Step 3 — Predictor tab

Train the AI load forecaster.

1. Check that Solar capacity (kWp) is set correctly for your site (0 if no solar).
2. Leave LightGBM rounds at 300 and Learning rate at 0.05 unless you have a reason to change.
3. Click **Train Forecaster**.
   - Training takes 30–90 seconds for a typical 2–3 month dataset.
   - A progress bar shows which horizon is being trained.
4. After training, a 24-hour forecast chart appears with P10 / P50 / P90 bands.
5. The trained model is saved to disk automatically.

**Scenario Adjustment Factor (optional):** If you know a specific event is coming
(e.g. a large production order tomorrow, or a site closure), enter a scaling factor
(e.g. 1.5 for 50% more load, 0.3 for 70% less). Set your confidence level. The adjusted
forecast is shown on a separate chart and fed to the live Manager automatically.

---

### Step 4 — AI Manager tab

Configure how the Manager handles the battery and controllable loads, then run it.

1. **Set parameters first** (top section):
   - Battery capacity and C-rate (pre-filled from Site Setup).
   - Peak target % — what fraction of the rolling 30-day reference peak to target
     (85% is a good starting point).
   - Charge upper threshold % — how high the site load can be while still allowing
     charging (70% is typical).
   - EV charger and HVAC settings with maximum cut percentages.

2. Click **Apply to live simulation** to activate these settings in the live view.

3. Click **Run on full history** to run the Manager over the entire uploaded dataset.
   This gives the Calculator and PowerRECO their reference data and takes 10–60 seconds
   depending on dataset size.

4. After the historical run completes, a before/after kVA chart and battery SOC chart appear.
   A downloadable CSV of all Manager decisions is available.

> **Note:** If you have the Live Simulation tab running, the Manager updates automatically
> every sim tick using the settings you applied. The live view shows three sections:
> what the Manager did this tick (actual decision on real data), the history of all
> executed decisions so far, and the forward plan based on the current forecast.

---

### Step 5 — Live Simulation tab (optional but recommended for demos)

This tab replays the last 30% of your uploaded data tick-by-tick, as if it were streaming
in real time.

1. Open the tab — initialisation takes about 10 seconds (the forecaster refits on the
   first 70% of data for an honest train/test split).
2. Use the **Play/Pause** button to advance automatically.
3. The forecast sharpens over time as more actuals arrive (warm-start retraining fires
   every 6 hours of simulated time).
4. The Manager tab updates alongside, showing live decisions.
5. The Calculator tab's **Live Monthly Bill Tracker** section updates automatically each
   time a full calendar month completes in the simulation.

---

### Step 6 — Bill Calculator tab

The Calculator works in two modes:

**Live Monthly Bill Tracker (top section):**
Updates automatically as the live simulation advances month by month.
Shows before/after bills, monthly savings, and a full component breakdown
(energy charge, MD charge, ICPT, KWTBB, Service Tax, NEM credit).
Use the month selector to drill into any specific month.

**Historical Batch Calculation (bottom section):**
Calculates the full bill across all uploaded months at once, using the Manager's historical
run from Step 4. Shows total before/after bills, monthly comparison chart, and a
downloadable CSV.

Switch the tariff schedule (sidebar) between **Post-July 2025 (RP4)** and **Pre-July 2025
(legacy)** for sensitivity analysis.

---

### Step 7 — PowerRECO tab

Runs after the Manager has processed the historical data.

1. Enter roof area (m²), panel wattage (default 415W), peak sun hours (default 4.5 for Malaysia),
   and cost inputs (solar RM/kWp, battery RM/kWh).
2. Click **Run PowerRECO Analysis**.

The tab shows results in this order:

- **Investment Feasibility** (top): Solar and battery each get a VIABLE / MARGINAL / NOT VIABLE
  verdict with estimated payback and a plain-English recommendation. If neither is viable, the
  analysis stops here to avoid misleading numbers.
- **Recommended System**: Solar kWp, panel count, battery kWh, MD reduction.
- **CAPEX Breakdown**: Solar panels, inverter, and battery costs listed separately with a pie chart.
- **25-Year Financial Return**: Payback, NPV, IRR, and annual savings split by component.
- **Monthly generation chart** and **energy flow analysis** in expandable sections.

---

## 7. File and folder structure

```
BOLT-bolt-integrated-ui/
│
├── BOLT_INTEGRATED/          ← Main application (run everything from here)
│   ├── app.py                ← Entry point: streamlit run app.py
│   ├── requirements.txt      ← All Python dependencies
│   ├── site_profiles.py      ← Pre-built site presets (SOL, E, SUN, MI2, Custom)
│   ├── persistence.py        ← Saves/loads site data between sessions
│   │
│   ├── calculator/           ← TNB tariff engine (7 tariff codes, RP4 + legacy)
│   ├── manager/              ← Battery dispatch and load curtailment optimizer
│   ├── predictor/            ← LightGBM probabilistic load forecaster
│   ├── powerreco/            ← Solar/battery sizing and 25-year ROI engine
│   ├── pipeline/             ← Data bridge between modules
│   │
│   └── data/                 ← Auto-created on first run (do not delete)
│       ├── site_overrides.json     ← Your Site Setup inputs, persisted to disk
│       ├── weather_cache/          ← Cached Open-Meteo weather data
│       └── sites/
│           └── <site_id>/
│               ├── load_profile.joblib   ← Your uploaded data
│               └── forecaster.joblib     ← Your trained model
│
├── Manager/                  ← Standalone Manager app (independent of main app)
├── PowerRECO/                ← Standalone PowerRECO app
├── Predictor/                ← Standalone Predictor app
└── SAM_CALCULATOR/           ← Standalone Calculator app
```

**Important:** Always run `streamlit run app.py` from inside the `BOLT_INTEGRATED` folder.
Running it from the repository root will cause import errors because the module paths
(e.g. `from calculator.tnb_tariffs import ...`) resolve relative to where you launch from.

---

## 8. Keeping the app running (longer sessions)

By default the app stops when you close the terminal. For longer sessions or demos:

**Keep the terminal open and minimised** — simplest option. The browser tab stays live.

**Run in the background (Windows PowerShell):**

```
Start-Job -ScriptBlock {
    Set-Location "C:\path\to\BOLT-bolt-integrated-ui\BOLT_INTEGRATED"
    & "C:\path\to\.venv\Scripts\streamlit.exe" run app.py
}
```

**Run in the background (macOS / Linux):**

```
nohup streamlit run app.py &> streamlit.log &
```

The app will continue running even if you close the terminal.
Check the log with `tail -f streamlit.log`. Stop it with `kill %1`.

**Change the default port** (if 8501 is already in use):

```
streamlit run app.py --server.port 8502
```

---

## 9. Troubleshooting

### "Module not found" or import errors

**Most likely cause:** you are running `streamlit run app.py` from the repository root instead of
the `BOLT_INTEGRATED` folder.

Fix: make sure you are in the right directory first.

```
cd BOLT_INTEGRATED
streamlit run app.py
```

---

### "Port 8501 is already in use"

Another instance of the app (or another Streamlit app) is already running on that port.

Option A — stop the other instance:

```
# Windows (find the process using port 8501)
netstat -ano | findstr :8501
taskkill /PID <the number shown> /F

# macOS / Linux
lsof -ti :8501 | xargs kill
```

Option B — run BOLT on a different port:

```
streamlit run app.py --server.port 8502
```

---

### Training is very slow

LightGBM uses all available CPU cores by default. On a machine with 4+ cores, training a
3-month dataset typically takes 30–90 seconds. If it takes longer:

- Reduce **LightGBM rounds** in the Predictor tab from 300 to 150. This halves training
  time with a small accuracy trade-off.
- Close other heavy applications to free CPU.

---

### Weather data fails to load

BOLT fetches real weather data from [Open-Meteo](https://open-meteo.com) — a free service,
no API key required. If the fetch fails (no internet, or Open-Meteo is temporarily down):

- The app falls back to a synthetic tropical weather model automatically.
- Forecast accuracy will be slightly lower but the app continues to work.
- Cached weather data (stored in `data/weather_cache/`) is reused on subsequent runs
  once fetched, so you only need internet on the first run.

---

### "Forecaster loaded but predict() failed" on startup

This happens when a previously saved model was trained on a different version of the code.

Fix: clear the saved model for that site.

1. Open the **Site Setup** tab.
2. Click **Clear saved data + model** for the affected site.
3. Re-upload your data and re-train the forecaster.

---

### The browser shows a blank white page

Streamlit sometimes takes a few extra seconds to compile on first launch. Wait 10 seconds
and refresh the browser tab. If it stays blank, check the terminal for error messages.

---

### Excel file fails to parse

BOLT supports all standard TNB meter export formats. If parsing fails:

- Make sure the file is not open in Excel at the same time.
- Try saving the file as `.csv` (File → Save As → CSV) and re-uploading.
- Check that the file has a timestamp column and at least one kW or kVAR column.

---

## 10. Updating to the latest version

To pull the latest code changes from GitHub:

```
# From the repository root (one level above BOLT_INTEGRATED)
git pull origin main
```

Then re-install dependencies in case anything new was added:

```
cd BOLT_INTEGRATED
pip install -r requirements.txt
```

Restart the app:

```
streamlit run app.py
```

Your saved data and trained models in the `data/` folder are not affected by updates.

---

## Quick-start cheat sheet

```
# 1. Clone (one-time)
git clone https://github.com/JohnDoeSKipper/BOLT-bolt-integrated-ui.git
cd BOLT-bolt-integrated-ui

# 2. Create and activate virtual environment (one-time)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install packages (one-time, or after updates)
cd BOLT_INTEGRATED
pip install -r requirements.txt

# 4. Run (every time)
streamlit run app.py
# Open browser → http://localhost:8501
```

---

*Maintained by Team: The King in The North*
*Ang Wei Jie, Jared · Soo Zi Ming · Lee Lin Sam*
