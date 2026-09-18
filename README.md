# Invoice-to-Payment Process Optimization: Throughput & Vendor Prediction

Process mining and machine learning applied to a SAP procurement dataset of a large multinational company headquartered in the Netherlands, containing 1,595,923 events across 251,734 cases ([BPI Challenge 2019](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/)) — combining process mining (PM4Py) and machine learning (scikit-learn, XGBoost) to discover processes, analyze throughput, check conformance and predict vendor performance.

## Executive Summary
- Analyzed [N] procurement cases from the BPI 2019 SAP event log using process mining and machine learning
- Discovered the as-is process with PM4Py (Inductive Miner) and identified [N] major bottlenecks/deviations through conformance checking
- Built a delay-prediction model ([algorithm], tuned with Optuna) achieving [F1 / precision / recall] on the test set
- Top predictors of delay: [feature 1], [feature 2], [feature 3] — explained via SHAP for individual-case transparency
- Practical use case: procurement teams can use the model's predictions and SHAP explanations to identify at-risk invoices early and prioritize intervention
- Business impact: reducing [specific bottleneck] could cut average case duration by [X] days, directly improving on-time payment rates and early-payment discount capture

*(Numbers above to be filled in once analysis is complete — this structure keeps the summary review-ready throughout the project.)*

## Project Overview

This project applies process mining and predictive modeling to a real-world SAP procurement event log provided by [BPI Challenge 2019](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/), covering the full pipeline from process discovery to a deployable prediction tool. The analysis follows the three classical process mining stages — **process discovery**, **conformance checking**, and **process enhancement** — and finally extends enhancement into predictive process monitoring using machine learning.

**Key steps:**
1. Load and clean the BPI 2019 event log (Notebook 1)
2. Process discovery — discover the as-is process with PM4Py (Notebook 2)
3. Conformance checking — identify deviations from the ideal process flow (Notebook 3)
4. Process enhancement — bottleneck and delay analysis using timestamp data (Notebook 4)
5. Feature engineering for delay and vendor-reliability prediction (Notebook 5)
6. Train and tune two predictive models — case-level throughput prediction (Part 1) and vendor Award tier prediction for existing and new vendors (Part 2, Models A & B) — using scikit-learn, XGBoost, and Optuna, tracked with Weights & Biases (W&B) (Notebook 6)
7. Explain model predictions with SHAP and dtreeviz (Notebook 7)

## Business Problem

The dataset originates from the [BPI Challenge 2019](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/): 
a large multinational company headquartered in the Netherlands, operating in the coatings and paints industry across 60 subsidiaries, submitted its purchase order handling process for investigation. The process owner's motivation was **compliance** — understanding not just how the process runs, but also where and how severely it deviates from expectation.

This project follows the three original questions posed by the BPI Challenge, extended with a predictive layer:

1. **Process discovery:** Is there a collection of process models that together properly describe the process captured in this data? (The challenge itself identifies at least four underlying flow types — 3-way matching with GR-based invoicing, 3-way matching without, 2-way matching and consignment.)
2. **Throughput analysis (enhancement):** What is the throughput of the invoicing process — the time between goods receipt, invoice receipt and payment (invoice clearing) — including matching the correct goods receipts to invoices when a single line item has several of each?
3. **Conformance and deviation:** Which purchase documents stand out from the log, where do they deviate from the discovered process models and how severe are these deviations — both in terms of process flow and invoice values (e.g. vendors producing disproportionate rework due to invoice errors)?
4. **Prediction (this project's extension):** Based on the findings in sections 1-3, three predictive models were developed addressing throughput and vendors' performance:
   - Part 1 — Case-Level Throughput Prediction: How long will a specific case take to clear?
   - Part 2 — Vendor Award Prediction: How reliable is an existing vendor (Model A; vendor-level) or a new vendor (Model B; order-level)? 

## Dataset

- **Source:** BPI Challenge 2019 — SAP procurement event log, published via [4TU.ResearchData](https://data.4tu.nl/articles/_/12715853/1) ([direct file download](https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828), 
  `BPI_Challenge_2019.xes`, ~695 MB uncompressed)
- **License:** CC BY 4.0 (dataset) — separate from this repository's MIT license, which covers code only. Citation: van Dongen, B.F., *BPI Challenge 2019*. 4TU.ResearchData.
- **Format:** IEEE XES standard, read natively via `pm4py.read_xes()`
- **Scope:** 1,595,923 events across 251,734 cases (case ID = purchase document + item), spanning 76,349 purchase documents, 42 activities and 627 users (607 human, 20 batch/automated) — covering purchase orders submitted in 2018 across 60 subsidiaries
- **Key attributes:** Case ID, activity, timestamp, resource (user), purchasing document ID, item type, item category (3-way with/without GR-based invoicing, 2-way, consignment), vendor, company (subsidiary), spend classification text, GR-based invoice verification flag, goods receipt flag
- **Note:** The raw `.xes` file is not committed to this repository due to its size — see the link above to download it directly. `data/raw/` is excluded via `.gitignore`.

## Approach

- **Process Mining:** PM4Py for process discovery (Inductive Miner) and
  conformance checking against the expected purchase-to-pay flow
- **Segmented analysis:** Where relevant, process discovery, conformance
  checking, and duration analysis are performed both in aggregate and
  segmented by item category (the four flow types), vendor, subsidiary
  and time period — since aggregate metrics can mask meaningful variation
  across these dimensions (e.g. a company-wide average duration can look
  acceptable while masking poor performance concentrated in a few
  subsidiaries)
- **Conformance checking:** BPI 2019 lacks a formal, machine-readable
  reference (de jure) process model, a known gap also faced by both
  winning BPI Challenge 2019 submissions
  ([Augusto, Leno & Reissner, 2019](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Student-Submission-1.pdf);
  [Diba, Remy & Pufahl, 2019](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Submission-6.pdf)) —
  this project checks conformance against two baselines instead: (1) a de
  facto model discovered from the log's dominant behavior and (2) a
  lightweight de jure reference following the challenge's documented flow
  (Purchase Order → Goods Receipt → Invoice Receipt → Clear Invoice)
- **Feature Engineering:** Three feature tables built for two predictive
  analyses (Notebook 5):
  - Part 1: target `gr_to_clear_days` (Goods Receipt → Clear Invoice in
    days, continuous), similar to
    [Rząd et al. (2019)](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Submission-2.pdf);
    prediction point restricted to information known as-of-Goods-Receipt
  - Part 2: target No Award / Bronze / Silver+ (3-class), based on
    `ir_to_clear_days` (Invoice Receipt → Clear Invoice in days); tiers
    are adapted from the UK's
    [Fair Payment Code](https://www.smallbusinesscommissioner.gov.uk/fpc/code-criteria/),
    with the Silver tier's small-business sub-criterion replaced by a
    stricter 90%-within-30-days threshold, and Silver merged with Gold
    into Silver+ to resolve a small-sample problem
- **Machine Learning:** Two predictive analyses (Notebook 6) — Part 1:
  case-level throughput prediction (champion model: XGBoost,
  Optuna-tuned, tracked in W&B), cross-validated against
  [Rząd et al. (2019)](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Submission-2.pdf);
  and Part 2: vendor Award tier prediction, using separate models for
  existing vendors (Logistic Regression, Decision Tree, both logged on
  W&B) and new/thin-history vendors (champion model: Optuna-tuned Random
  Forest, logged on W&B)
- **Explainability:** SHAP for global and individual-case feature
  importance, complemented by a dtreeviz visualization of a
  representative decision tree for structural interpretability, applied
  to both champion models

## Planned Extensions

These extensions are committed and will be completed — the open question
is timing, not whether. They are deliberately decoupled from the
September 20, 2026 deadline so they do not compete with the core pipeline
under time pressure.

- **Social network analysis** — resource collaboration patterns via
  PM4Py's handover-of-work and working-together networks, cross-referenced
  with duration data to distinguish genuine bottlenecks from
  high-throughput specialists
- **Object-centric process mining** — OCEL 2.0 conversion and PM4Py's
  object-centric discovery, to more accurately capture the one-to-many
  relationships (e.g., multiple goods receipts/invoices per line item)
  that this project's single-case-notion analysis simplifies
- **Power BI Dashboard & Process.Science Integration** — a three-dashboard
  Power BI application (Process Overview, Vendor & Spend, Rework &
  Bottlenecks) incorporating this project's model predictions, alongside a
  demonstration of Process.Science's commercial process-mining visual

*See [`docs/planned-extensions.md`](docs/planned-extensions.md) for
detailed methodology, specific research questions, and citations for each
extension.*

## Key Findings — Core Analysis

*(To be completed once analysis is finished.)*

## Business Recommendations

## Limitations & Further Research

## Tools & Technologies

PM4Py · scikit-learn · Optuna · W&B · SHAP · dtreeviz · Power BI (incl. native Python visual integration) · Process.Science (Power BI visual) · Disco · psmineR (R, stretch goal) · Power Automate Process Mining visual (stretch goal) · Claude (Anthropic)

## Repository Structure

```
invoice-to-payment-process-optimization/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_process_discovery.ipynb
│   ├── 03_conformance_checking.ipynb
│   ├── 04_process_enhancement.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_throughput_and_vendor_prediction.ipynb
│   └── 07_shap_dtreeviz_explainability.ipynb
│
├── images/
│   ├── 02_process_discovery/
│   │   ├── petri_net_3way_before_gr.png
│   │   └── petri_net_3way_before_gr_no_srm.png
│   ├── 03_conformance_checking/
│   │   ├── de_facto_model.png
│   │   └── de_jure_model.png
│   ├── 04_process_enhancement/
│   │   └── activity_bottleneck_by_theme.png
│   ├── 06_throughput_and_vendor_prediction/
│   │   ├── part1_predicted_vs_actual.png
│   │   ├── part1_xgboost_feature_importance.png
│   │   ├── model_a_confusion_matrices.png
│   │   ├── model_b_confusion_matrix.png
│   │   └── model_b_feature_importance.png
│   └── 07_shap_dtreeviz_explainability/
│       ├── part1_shap_summary.png
│       ├── part1_dtreeviz.png
│       ├── model_b_shap_summary.png
│       └── model_b_dtreeviz.png
│
└── docs/
    └── planned-extensions.md
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
