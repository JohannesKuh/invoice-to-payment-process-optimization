# Invoice-to-Payment Process Optimization & Delay Prediction

Process mining and machine learning applied to SAP procurement data (BPI Challenge 2019) — combining PM4Py, scikit-learn, and Power BI to discover process bottlenecks and predict invoice payment delays.

## Executive Summary
- Analyzed [N] procurement cases from the BPI 2019 SAP event log using process mining and machine learning
- Discovered the as-is process with PM4Py (Inductive Miner) and identified [N] major bottlenecks/deviations through conformance checking
- Built a delay-prediction model ([algorithm], tuned with Optuna) achieving [F1 / precision / recall] on the test set
- Top predictors of delay: [feature 1], [feature 2], [feature 3] — explained via SHAP for individual-case transparency
- Practical use case: procurement teams can use the model's predictions and SHAP explanations to identify at-risk invoices early and prioritize intervention
- Business impact: reducing [specific bottleneck] could cut average case duration by [X] days, directly improving on-time payment rates and early-payment discount capture

*(Numbers above to be filled in once analysis is complete — this structure keeps the summary review-ready throughout the project.)*

## Project Overview

This project applies process mining and predictive modeling to a real-world SAP procurement event log, covering the full pipeline from process discovery to a deployable prediction tool. The analysis follows the three classical process mining stages — **process discovery**, **conformance checking**, and **process enhancement** — and finally extends enhancement into predictive process monitoring using machine learning.

### Key Steps
1. Load and clean the BPI 2019 event log (Apache Hop / pandas)
2. Process discovery — discover the as-is process with PM4Py (Inductive Miner)
3. Conformance checking — identify deviations from the ideal process flow
4. Process enhancement — bottleneck and delay analysis using timestamp data
5. Feature engineering for delay prediction
6. Train and tune a delay-prediction model (scikit-learn + Optuna), tracked with MLflow
7. Explain model predictions with SHAP
8. Build a Power BI dashboard for business-facing KPIs, including a native Python-visual embedding of PM4Py directly inside Power BI
9. Demonstrate the same process discovery step using an industry-standard commercial tool (Process.Science / Disco) alongside the open-source pipeline
10. *(Stretch goal)* Performance spectrum analysis using R's psmineR package, exploring segment-level timing patterns as a complement to the standard bottleneck analysis
11. *(Stretch goal)* Side-by-side comparison of free Power BI process-mining visuals (Process.Science, Microsoft's native Power Automate Process Mining visual) on the same data subset

## Business Problem

Late invoice payments and inefficient procurement processes carry real costs: missed early-payment discounts, strained supplier relationships, and manual rework when exceptions occur. This project asks two linked business questions:

1. **Where** in the invoice-to-payment process do delays and deviations actually occur (as opposed to where the documented process assumes they occur)?
2. **Can delays be predicted early enough** — e.g., right after goods receipt or invoice entry — for procurement teams to intervene before payment is late?

The goal is a combination of diagnostic insight (process mining) and a predictive layer (machine learning) that together support both process redesign and day-to-day case triage.

## Dataset

- **Source:** BPI Challenge 2019 — SAP procurement event log, [4TU Data Repository](https://data.4tu.nl/)
- **License:** Dataset is released under its own terms (CC-BY or similar per 4TU) — separate from this repository's MIT license, which covers code only
- **Scope:** [N] cases, [N] events, covering the purchase-to-pay process from purchase order through invoice and payment
- **Key attributes:** case ID, activity, timestamp, resource, purchasing document type, vendor, item category

## Approach

- **Process Mining:** PM4Py for process discovery (Inductive Miner) and conformance checking against the expected purchase-to-pay flow
- **ETL:** Apache Hop for reproducible event-log extraction and transformation
- **Machine Learning:** delay-prediction model (scikit-learn), tuned via Optuna, tracked in MLflow
- **Explainability:** SHAP for global and individual-case feature importance
- **Visualization:** Power BI dashboard for business-facing KPIs, built on model predictions scored offline and exported as a table, including a native
  Python-visual embedding of PM4Py (via [viadee's](https://www.viadee.de/en/blog/process-mining-mit-power-bi/) approach)
- **Industry Tool Demonstration:** limited demos of Process.Science's free Power BI visual and Disco (30-day trial), cross-validating the open-source pipeline against commercial tools on a filtered data subset
- **Stretch goals:** R-based performance spectrum analysis ([psmineR](https://cran.r-project.org/web/packages/psmineR/index.html)); side-by-side comparison of free Power BI process-mining visuals

## Key Findings

*(To be completed once analysis is finished.)*

## Tools & Technologies

PM4Py · Apache Hop · scikit-learn · Optuna · MLflow · SHAP · Power BI (incl. native Python visual integration) · Process.Science (Power BI visual) · Disco · psmineR (R, stretch goal) · Power Automate Process Mining visual (stretch goal) · Claude (Anthropic)

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
│   ├── invoice_dashboard.pbix              # includes native Python visual (PM4Py)
│   └── screenshots/
│       └── process_science_discovery.png     # Process.Science demo (see Approach)
│
├── reports/
│   ├── process_mining_findings.md
│   └── model_performance_summary.md
│
└── docs/
    └── architecture_diagram.png      # PM4Py → ML → Power BI flow
```

## Acknowledgements

- Dataset provided ...
- Certificate: ... 🎓 (TU/e/Coursera)
- AI assistance provided by Claude (Anthropic) for code guidance, 
  interpretation refinement and documentation support

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. Note: the MIT license applies to the code in this repository only. The BPI 2019 dataset is governed by its own license from the 4TU Data Repository.

## Author

Johannes Kuhaupt, LL.M., PMP
[LinkedIn](https://www.linkedin.com/in/johanneskuhaupt/) · [GitHub](https://github.com/JohannesKuh/)
