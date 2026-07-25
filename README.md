# Invoice-to-Payment Process Optimization & Delay Prediction

Process mining and machine learning applied to SAP procurement data (BPI Challenge 2019) — combining PM4Py, scikit-learn, and Power BI to discover process bottlenecks and predict invoice payment delays.

## Executive Summary
- Analyzed [N] procurement cases from the BPI 2019 SAP event log using process mining and machine learning
- Discovered the as-is process with PM4Py (Inductive Miner) and identified [N] major bottlenecks/deviations through conformance checking
- Built a delay-prediction model ([algorithm], tuned with Optuna) achieving [F1 / precision / recall] on the test set
- Top predictors of delay: [feature 1], [feature 2], [feature 3] — explained via SHAP for individual-case transparency
- Practical use case: a Streamlit "what-if" app lets procurement teams test how process changes (e.g., automated matching, earlier goods receipt) shift predicted delay risk
- Business impact: reducing [specific bottleneck] could cut average case duration by [X] days, directly improving on-time payment rates and early-payment discount capture

*(Numbers above to be filled in once analysis is complete — this structure keeps the summary review-ready throughout the project.)*

## Project Overview

This project applies process mining and predictive modeling to a real-world SAP procurement event log, covering the full pipeline from process discovery to a deployable prediction tool.

### Key Steps
1. Load and clean the BPI 2019 event log (Apache Hop / pandas)
2. Discover the as-is process with PM4Py (Inductive Miner)
3. Conformance checking — identify deviations from the ideal process flow
4. Bottleneck and delay analysis
5. Feature engineering for delay prediction
6. Train and tune a delay-prediction model (scikit-learn + Optuna), tracked with MLflow
7. Explain model predictions with SHAP
8. Build a Power BI dashboard for business-facing KPIs
9. Build a Streamlit "what-if" app for scenario testing
10. Demonstrate the same process discovery step using an industry-standard commercial tool (Process.Science / Disco) alongside the open-source pipeline

## Business Problem

Late invoice payments and inefficient procurement processes carry real costs: missed early-payment discounts, strained supplier relationships, and manual rework when exceptions occur. This project asks two linked business questions:

1. **Where** in the invoice-to-payment process do delays and deviations actually occur (as opposed to where the documented process assumes they occur)?
2. **Can delays be predicted early enough** — e.g., right after goods receipt or invoice entry — for procurement teams to intervene before payment is late?

The goal is a combination of diagnostic insight (process mining) and a forward-looking tool (ML + Streamlit) that together support both process redesign and day-to-day case triage.

## Dataset

- **Source:** BPI Challenge 2019 — SAP procurement event log, [4TU Data Repository](https://data.4tu.nl/)
- **License:** Dataset is released under its own terms (CC-BY or similar per 4TU) — separate from this repository's MIT license, which covers code only
- **Scope:** [N] cases, [N] events, covering the purchase-to-pay process from purchase order through invoice and payment
- **Key attributes:** case ID, activity, timestamp, resource, purchasing document type, vendor, item category

## Approach

- **Process Mining:** PM4Py for process discovery (Inductive Miner) and conformance checking against the expected purchase-to-pay flow
- **ETL:** Apache Hop for reproducible event-log extraction and transformation ahead of analysis
- **Machine Learning:** delay-prediction model trained with scikit-learn, hyperparameters tuned via Optuna, experiments tracked in MLflow
- **Explainability:** SHAP values for both global feature importance and individual-case explanations
- **Visualization:** Power BI dashboard for business-facing KPIs (case duration, delay rate, bottleneck locations)
- **Interactivity:** Streamlit app for what-if scenario testing on individual cases
- **Industry Tool Demonstration:** In addition to the open-source pipeline, this project includes a limited demonstration of two commercial process-mining tools:
  - **Process.Science's free Power BI visual** (Microsoft AppSource) — shows direct process-mining integration inside Power BI on a filtered subset of the event log
  - **Disco** (Fluxicon, 30-day trial) — used to cross-validate the discovered process map against PM4Py's output
  
  These are included to demonstrate familiarity with commercial tooling common in enterprise process-mining roles, and to make an explicit, honest case for why the core pipeline stays open-source and reproducible. Due to free-tier data limits, these demonstrations use a filtered subset rather than the full BPI 2019 log.

## Key Findings

*(To be completed once analysis is finished.)*

## Tools & Technologies

PM4Py · Apache Hop · scikit-learn · Optuna · MLflow · SHAP · Power BI · Streamlit · Process.Science (Power BI visual) · Disco

## Repository Structure

```
invoice-to-payment-process-optimization/
│
├── README.md
├── LICENSE                          # MIT (code only — see Dataset section for data license)
├── requirements.txt / environment.yml
├── .gitignore
│
├── data/
│   ├── raw/                         # BPI 2019 XES/CSV (or link if too large for repo)
│   ├── processed/                   # cleaned event log, feature-engineered CSVs
│   └── data_dictionary.md
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_process_discovery_pm4py.ipynb      # Inductive Miner, process map
│   ├── 03_conformance_checking.ipynb         # deviations, bottlenecks
│   ├── 04_feature_engineering.ipynb          # case duration, delay labels, etc.
│   ├── 05_delay_prediction_model.ipynb       # scikit-learn + Optuna tuning
│   └── 06_shap_explainability.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── process_mining_utils.py
│   ├── model_training.py
│   └── evaluation.py
│
├── mlflow/
│   └── (tracking config / experiment notes, not the full mlruns/ folder)
│
├── powerbi/
│   ├── invoice_dashboard.pbix
│   └── screenshots/
│       └── process_science_discovery.png     # Process.Science demo (see Approach)
│
├── streamlit_app/
│   ├── app.py                       # "what-if" delay prediction app
│   └── requirements.txt
│
├── reports/
│   ├── process_mining_findings.md
│   └── model_performance_summary.md
│
└── docs/
    └── architecture_diagram.png      # PM4Py → ML → Power BI/Streamlit flow
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. Note: the MIT license applies to the code in this repository only. The BPI 2019 dataset is governed by its own license from the 4TU Data Repository.

## Author

Johannes Kuhaupt, LL.M., PMP
