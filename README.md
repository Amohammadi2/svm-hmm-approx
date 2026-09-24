# Stochastic Market Modeling & VaR Analytics Platform

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Selenium](https://img.shields.io/badge/Scraping-Selenium-green.svg)

An end-to-end Python application for turning live web-scraped financial data into a reproducible statistical modeling and **Value at Risk (VaR)** workflow.

The project combines practical data acquisition, resilient browser automation, local data persistence, parameter estimation, predictive risk analysis, and interactive visualization in a single Streamlit application. The emphasis is on making the complete analytical path visible: from the source webpage and raw observations to fitted model parameters and out-of-sample risk violations.

The application currently works with historical gold-price data published by **TGJU**, using the [18K gold historical-price table](https://www.tgju.org/profile/tgju_gold_irg18/history) as its data source.

---

## What the project does

The application exposes the modeling workflow through a Streamlit interface:

```text
TGJU Web Table
      │
      ▼
Resilient Selenium Scraper
      │
      ▼
Raw CSV / JSON Cache
      │
      ▼
DataFrame Preview & Processing
      │
      ▼
Parameter Fitting
      │
      ├── Hyperparameter / fitting settings
      │
      ▼
Fitted-Parameter Cache
      │
      ▼
One-step-ahead Risk Prediction
      │
      ▼
VaR Dashboard
      ├── Configurable confidence level (α)
      ├── Actual violation rate
      └── Log returns vs. VaR + violation zones
```

This is intentionally a **single-asset research and analytics application**, rather than a distributed trading platform. The current implementation prioritizes clarity, reproducibility, modularity, and the ability to inspect each stage of the pipeline.

---

## Key Features

### Interactive Streamlit application

The project is no longer a command-line-only computational pipeline. Streamlit provides a user-facing interface for the entire workflow:

- scrape or reuse cached market data
- preview the resulting DataFrame
- configure parameter-fitting and hyperparameter settings
- fit the stochastic model
- reuse cached fitted parameters
- calculate predictive VaR
- select the VaR confidence level, `α`
- inspect the **realized violation rate**
- visualize daily log returns against VaR predictions and the corresponding violation zones

The goal is to make statistical modeling inspectable rather than hiding the entire process behind a single script.

### Parameter estimation

The inference layer contains the parameter-estimation engine in **pure Python**.

Fitting is exposed through configurable settings so that model behavior can be investigated without changing the application architecture. Fitted parameters can also be cached, avoiding unnecessary repeated optimization when the underlying inputs and fitting configuration have not changed.

### Value at Risk (VaR) analytics

The application turns the fitted model into an operational risk-analysis dashboard.

For a selected confidence level `α`, the dashboard presents the model's VaR predictions alongside the realized log returns and highlights observations that cross the risk threshold.

It also reports the **observed violation rate** so the model can be evaluated against what actually happened in the historical sample rather than relying only on a visual comparison.

Conceptually, the workflow is:

```math
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) \times 100
```

where `P_t` is the daily closing price and `r_t` is the daily percentage log return.

The dashboard then compares realized returns with the model's predicted lower-tail risk threshold.

### Cached data layer

The project does not require a database server.

Instead, the `datalink` layer persists intermediate results using simple **CSV and JSON files**, including:

- scraped market data
- processed data used by the inference layer
- fitted model parameters and related cached results

This keeps the application lightweight and easy to run locally or inside Docker while making intermediate artifacts inspectable and reproducible.

---

## Resilient Web Scraping

One of the less visible but technically important parts of the project is the scraper.

The source website uses an **AJAX-updated, interactive data table**, which means the DOM can be re-rendered while Selenium is interacting with it. A naïve scraper can therefore fail even when the page appears to be loaded correctly.

The scraper is designed around this failure mode. It handles problems such as:

- stale element references after JavaScript re-rendering
- elements becoming invalid between lookup and interaction
- pagination/interactivity issues caused by asynchronous DOM updates
- repeated element acquisition when previously captured references are no longer valid

Rather than assuming a static HTML page, the scraper treats the browser interface as a changing state and recovers from common Selenium synchronization failures.

This makes the data-acquisition layer more robust against the particular behavior of the source website.

---

## Architecture

The repository is deliberately modular. The root package is divided into four main responsibilities:

```text
/
├── app/
│   └── Streamlit application
│
├── inference/
│   └── Parameter estimation and VaR calculation
│
├── datalink/
│   └── Scraping, persistence, loading, and data access
│
├── utils/
│   └── Shared utilities
│
└── notebooks/
    └── Parameter-recovery and exploratory notebooks
```

### `app/`

Contains the Streamlit application and its presentation-layer logic.

The application is responsible for connecting user controls and visual outputs to the underlying data and inference modules without putting the statistical implementation directly inside the UI code.

### `inference/`

Contains the statistical computation layer:

- parameter estimation
- fitted-model handling
- predictive risk calculations
- VaR generation

This separation makes the modeling code usable independently of the Streamlit interface.

### `datalink/`

Owns the boundary between the application and external data.

It handles:

- Selenium-based acquisition from TGJU
- raw-data persistence
- loading cached data
- saving/loading fitted results where applicable

Keeping this boundary separate from the inference engine means the statistical layer does not need to know how the observations were obtained.

### `utils/`

Contains reusable supporting functionality shared across the project.

---

## Parameter Recovery Lab

The repository includes a **parameter recovery lab** in:

```text
/notebooks
```

The notebook environment is intended for people who want to go beyond the application UI and experiment directly with the statistical machinery.

This is useful for:

- understanding the parameterization
- testing whether known parameters can be recovered from simulated or controlled data
- exploring estimator behavior
- debugging modeling assumptions
- experimenting with the inference code interactively

The notebooks are therefore not just documentation; they provide an experimental workspace around the core model.

---

## Why the project is Python-only

The earlier implementation used a separate C++ estimation engine. That architecture has been removed.

The current project is **pure Python**, including the inference layer.

This simplifies:

- development and debugging
- deployment
- reproducibility
- dependency management
- movement between notebooks and the application

It also makes the statistical code directly accessible to Python's data-analysis ecosystem instead of requiring a language boundary between data preparation, model fitting, and visualization.

---

## Storage: Simple Files Instead of SQL

The project deliberately uses local **CSV and JSON artifacts** rather than a relational database.

That choice matches the current scope of the application: a single-asset analytical workflow where the main requirement is reproducible persistence of intermediate results, not multi-user transactional querying.

The result is a small operational footprint:

```text
No database server
No ORM
No SQL schema
No external queue
No distributed worker layer
```

The persisted files are also easy to inspect, archive, version, and reproduce during research or analysis.

---

## Dockerized Execution

The application is containerized and can be run by building its Docker image.

Docker packages the application environment, Python dependencies, browser automation stack, and runtime configuration into a reproducible execution environment.

The image is configured to obtain project dependencies through **Iranian software/package mirrors**, with the exception of the Selenium browser driver: **Firefox Geckodriver** is obtained separately rather than through those mirrors.

This setup is useful when deployment needs to remain reproducible despite local dependency and network constraints.

---

## Technical Focus

Although the interface is small, the project spans several practical areas that are common in real data-science work:

**Data acquisition**

The system extracts structured historical observations from a dynamic financial website whose table is updated through JavaScript.

**Data engineering**

Scraped observations are normalized, persisted, reloaded, and passed cleanly between the acquisition and inference layers.

**Statistical modeling**

The inference layer estimates model parameters and exposes the fitting configuration needed to investigate model behavior.

**Predictive risk analysis**

The fitted model is used to generate one-step-ahead VaR predictions and compare them with realized returns.

**Model evaluation**

The realized violation rate provides a direct diagnostic of how frequently the observed returns crossed the predicted risk threshold.

**Data visualization**

Matplotlib is used to show the relationship between realized log returns, predicted VaR, and violation regions.

**Reproducibility**

Raw data and fitted results are cached as inspectable files, while the full environment can be reproduced through Docker.

---

## Running the Project

Build the Docker image from the repository root:

```bash
docker build -t stochastic-market-var .
```

Then run the resulting container according to the port configuration defined by the project's Docker setup.

Once the Streamlit application is running, the workflow is:

```text
1. Acquire or load cached TGJU data
2. Inspect the DataFrame
3. Configure parameter fitting
4. Fit or load cached parameters
5. Configure VaR confidence level α
6. Generate VaR predictions
7. Inspect the realized violation rate
8. Analyze the Matplotlib return/VaR visualization
```

For experimentation outside the web interface, the notebooks in `/notebooks` provide direct access to the underlying code.

---

## Data Source

Historical market data is collected from TGJU:

**18K Gold — Historical Data**  
https://www.tgju.org/profile/tgju_gold_irg18/history

The repository is therefore best understood as a **research-oriented single-asset analytics system**, currently demonstrated on the Iranian 18K gold market.

---

## License

This project is released under the MIT License.
