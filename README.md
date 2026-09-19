# Invoice-to-Payment Process Optimization: Throughput & Vendor Prediction

Process mining and machine learning applied to a SAP procurement dataset of a large multinational company headquartered in the Netherlands, containing 1,595,923 events across 251,734 cases ([BPI Challenge 2019](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/)) — combining process mining (PM4Py) and machine learning (scikit-learn, XGBoost) to discover processes, analyze throughput, check conformance and predict vendor performance.

## Executive Summary

- Applied process mining and predictive modeling to a real SAP procurement
  event log from a multinational company (BPI Challenge 2019),
  covering 1,595,923 events across 251,734 cases
- Discovered that a single process model cannot describe this process —
  at least four segmented models are required by item category, since
  the two dominant flow types ("3-way match, invoice before GR", 77.37% of events, and "3-way match, invoice after GR", 20.00% of events) produce unreadable
  "spaghetti" models even after excluding known sub-populations
- Found median end-to-end invoice clearing takes **63 days**, driven
  primarily by the Invoice Receipt → Clear Invoice stage (accounting for
  roughly two-thirds of the median total), with throughput varying mainly by category, activity and vendors (e.g. throughput time between the fastest and slowest vendors varies **20x**, requiring a targeted process improvement analysis)
- Trained and tuned **three predictive models**: Case-level throughput
  prediction (champion: tuned XGBoost, RMSE = 20.05 days, R² = 0.574,
  Part 1) and vendor reliability classification (No Award / Bronze /
  Silver+) for existing and new vendors (Part 2, Models A and B; Model
  B's champion: Random Forest, cross-validated macro F1 = 0.729), both
  cross-validated against independent published BPI submissions
- Model verification with SHAP & dtreeviz: Analysis independently
  confirmed the feature importance of each champion model (Part 1: tuned
  XGBoost, Part 2, Model B: Random Forest) — for example,
  `spend_classification_NPR` and `order_value` account for 58% of Model
  B's total feature importance — three independent explainability
  methods providing strong evidence of the discovered patterns
- Business recommendations: Applying the tuned XGBoost throughput model
  (Part 1) to open cases identifies poor-performing vendor groups
  needing closer attention (e.g. "Insufficient Data" vendors, ~106 days
  to clear on average); applying Model B's champion Random Forest to
  vendor-tier prediction demonstrates confident, correct predictions
  (>99% top probability) in clear-cut cases, while also revealing when a
  prediction is a genuine close call that should be manually verified
  rather than used for automatic classification
- Business impact: Since over 60% of total procurement value results from
  operational efforts, according to [Bain & Company (2022)](https://www.bain.com/insights/how-to-succeed-in-procurement-in-2022/), combining process mining and machine learning helps identify real
  operational deficiencies — e.g. complex process flows, poor vendor
  performance — and predict them early, enabling active management decisions to improve overall procurement outcomes

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

*For the complete analysis, methodology, and supporting data tables, see
the relevant notebook — section references are noted throughout below.*

**1. Process discovery:** Segmenting by `case:Item Category` reveals the
root cause of BPI 2019's known complexity — the two dominant flow types,
**Process 1 ("3-way match, invoice before GR", 77.37% of events)** and
**Process 2 ("3-way match, invoice after GR", 20.00% of events)**, together
account for 97.4% of all activity yet produce unreadable "spaghetti"
models even after excluding known sub-populations like SRM cases, while
the two smaller categories are clean and interpretable. This confirms the
challenge's own suggestion that item properties should determine which of
(at least) four required models applies (Notebook 2).

![3-way match, invoice before GR](images/02_process_discovery/petri_net_3way_before_gr.png)

*Figure 1: The dominant flow type (77% of events) produces an unreadable
"spaghetti" model — visual confirmation that a single unsegmented model
cannot describe this process.*

**2. Throughput analysis (enhancement):** The median throughput across each stage of
the invoicing process is:

- **GR → Invoice Receipt:** 9.14 days (n = 210,370)
- **Invoice Receipt → Clear Invoice:** 42.05 days (n = 183,293)
- **GR → Clear Invoice (end-to-end):** 63.00 days (n = 182,808)

The **main driver** of end-to-end throughput is the Invoice Receipt →
Clear Invoice stage, accounting for roughly two-thirds of the median
total.

*Note on methodology:* throughput is calculated using simplified
first-occurrence matching — the first Goods Receipt event is matched to
the first Invoice Receipt event, and so on. The challenge's deeper
question — precisely matching *multiple* GR and invoice messages within a
single line item, where several of each can occur — is deferred to the
planned Object-Centric Process Mining (OCPM) extension.

This 63-day aggregate figure masks substantial heterogeneity:

- **By category:** Segmenting the three throughput deltas (GR→IR,
  IR→Clear, GR→Clear) by `case:Item Category` shows the two dominant flow
  types — **"3-way match, invoice before GR"** (77.4% of events) and
  **"3-way match, invoice after GR"** (20.0% of events) — converge on a
  similar ~63-day end-to-end total, but via opposite internal patterns.
  "Invoice after GR" is *front-loaded* — most of its delay happens early,
  in the GR→IR stage (26 days) — while "invoice before GR" is
  *back-loaded*, with GR→IR taking only 9 days but IR→Clear stretching to
  43 days. Two categories can share the same headline number while having
  entirely different underlying bottlenecks.
- **By vendor:** Throughput varies over 20x between the fastest and
  slowest vendors, and *where* the delay occurs also differs by vendor —
  e.g., `vendor_0135` is the single fastest vendor at the GR→IR stage
  (2 days) but the *slowest* overall (111 days end-to-end), since its
  entire delay concentrates in the later IR→Clear stage. A vendor that
  looks fast at one checkpoint can still be the worst performer overall.
- **By activity:** Excluding SRM cases, five recurring patterns emerged
  among the top 20 transitions by occurrence, and the longest-duration
  transitions among pairs occurring at least 50 times (186 of 383 pairs).
  The core process flow confirms "Record Invoice Receipt" → "Clear
  Invoice" (133,595 occurrences, median 36.19 days) as the single primary
  bottleneck — direct, activity-level confirmation that IR→Clear is the
  main driver of end-to-end throughput.

![Activity-Level Bottleneck Analysis by Theme](images/04_process_enhancement/activity_bottleneck_by_theme.png)

*Figure 2: Five recurring patterns emerge across the process's top
transitions — approval-related delays, the core process flow, a
deviation cluster, a payment-block sub-flow, and repetitive self-loop
activities — mapping directly onto where and why cases actually slow
down.*

**3. Conformance and deviation:** Conformance depends heavily on the reference model and level of
aggregation used — de facto, de jure, vendor, and document views each
surface different, complementary insights (Notebook 3). Two of the three
findings below compare directly against the de jure model shown here;
document-level deviation is assessed against the de facto model instead.

![De jure model](images/03_conformance_checking/de_jure_model.png)

*Figure 3: The de jure model — the documented core sequence — fits only
67% of cases exactly, confirming that each flow type requires its own
reference model.*

- **Which purchase documents stand out?** Several documents show 97-100%
  of their line items deviating (e.g. `4507021416`: 172/172 items, 100%).
  Documents follow an even more extreme long tail than vendors (max: 429
  items), yet only 1 of the top 10 deviating documents also appears among
  the top 10 largest overall — confirming document-specific factors, not
  size, drive these deviations.
- **Where are deviations, and how severe?** An unfiltered de facto model
  shows 99.6% fitness, while the documented de jure policy shows only 67%
  — each flow type genuinely needs its own reference model.
  Cross-referencing deviator groups against de jure rules reveals **two
  striking, counterintuitive patterns**: violations are nearly 5x higher
  among general deviators versus baseline at the final invoice-clearing
  step — suggesting this is where genuine process friction happens
  (disputes, manual intervention, blocked payments) — while
  high-multiplicity deviators are the *most* compliant group on every
  single rule, well below baseline — indicating these cases deviate due
  to structural complexity (multiple GR/invoice objects per case), not
  genuine non-compliance. A single-case-notion model cannot represent
  this distinction cleanly, directly motivating the planned OCPM
  extension.

  | Model / Group | PO Item → GR | GR → Invoice Receipt | Invoice Receipt → Clear Invoice |
  |---|---|---|---|
  | Baseline (all cases) | 6.58% | 10.94% | 13.16% |
  | All de facto deviators (8,486) | 5.89% | 19.60% | 64.00% |
  | High-multiplicity deviators (2,349) | 1.45% | 6.48% | 7.58% |

- **Which vendors produce disproportionate rework?** Among the top 100
  vendors by volume, 26 unique vendors appear across the three "worst 10"
  lists, with violation rates (24-38%) substantially above baseline. For instance,
  `vendorID_0282` shows a genuine 100% violation rate on invoice clearing
  (277/277 cases). Notably, the two highest-volume vendors overall
  (`vendorID_0136`, `vendorID_0120`, 13,000+ cases each) show only
  moderate violation rates (2.97-23.23% and 1.79-20.54%) — indicating
  **business volume does not drive violations; the worst rates
  concentrate among mid-sized vendors.**

**4. Prediction (this project's extension, Notebook 6):**

Based on these findings, three models were developed predicting throughput
(Part 1) and vendors' performance (Part 2):

- **Part 1 — Case-Level Throughput Prediction:** The **champion model
  tuned XGBoost** achieves RMSE 20.05 days (R² 0.574) on the test set,
  outperforming Linear Regression, Decision Tree, and Random Forest
  baselines, and cross-validated against
  [Rząd et al. (2019)](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Submission-2.pdf).
  A prior study's top-cited predictor was found to rely on information
  not actually knowable at this project's stricter, genuinely
  forward-looking prediction point. Feature importance is dominated by a
  single spend category (`sub_spend_area_Labels`) — cases in this
  category take a consistent 33.5-day-longer median throughput than all
  others, a concrete, actionable pattern. Performance also varies
  meaningfully by vendor reliability tier, with Gold-tier vendors' cases
  predicted most accurately and Silver-tier the least.

- **Part 2 — Vendor Award Prediction:** Two complementary models predict
  a vendor's reliability tier (No Award/Bronze/Silver+) — **Model A** for
  existing vendors (Logistic Regression: macro F1 0.455, Decision Tree:
  macro F1 0.416, constrained by only 445 rated vendors, no single
  champion given their distinct error profiles) and **Model B** for
  new/thin-history vendors (macro F1 0.50 on a held-out test set,
  cross-validated macro F1 0.729, trained on 166,447 transactions). A
  significant methodological finding — `XGBClassifier` silently ignoring
  `class_weight` — was caught and corrected during development, revealing
  **Random Forest** as the **champion model**. **Given each model's real
  limitations, predictions are intended to support — not replace — manual
  review**, especially for close calls between adjacent tiers.

![Model B confusion matrix](images/06_throughput_and_vendor_prediction/model_b_confusion_matrix.png)

*Figure 4: Model B's confusion matrix shows that errors concentrate almost
entirely between adjacent tiers — a coherent, ordered sense of vendor
reliability, even where exact boundaries remain uncertain.*

**SHAP & dtreeviz Explainability (Notebook 7):** Three independent
methods (built-in feature importance, SHAP, and tree structure)
consistently agree on each model's dominant drivers — `vendor_tier_No
Award` and `sub_spend_area_Labels` for Part 1; `spend_classification_NPR`
and `order_value` for Model B — providing strong evidence that these
patterns reflect genuine structure in the data, not artifacts of any
single measurement approach.

![dtreeviz structural visualization (Part 1)](images/07_shap_dtreeviz_explainability/part1_dtreeviz.png)

*Figure 5: A representative tree from the XGBoost ensemble (Part 1) shows
its very first split is on `vendor_tier_No Award` — the model's single
most important decision — directly confirming the same feature identified
as dominant by two independent methods.*

![Model B SHAP summary](images/07_shap_dtreeviz_explainability/model_b_shap_summary.png)

*Figure 6: Model B's SHAP summary independently confirms
`spend_classification_NPR` and `order_value` as the dominant drivers of
vendor tier predictions, consistently across all three award tiers.*

## Business Recommendations

The trained models were applied to different business scenarios based on
the procurement dataset. The outcomes lead to the following business
recommendations:

**Part 1 — Throughput monitoring:** Applying the champion model to the
38,047 currently open cases shows predicted clearing times ranging from
~32 days (Gold-tier vendors) to ~106 days on average (Insufficient Data
vendors). Two groups require closer monitoring: **"Insufficient Data"
vendors** (slowest predicted times, likely reflecting less-established
relationships or weaker internal invoicing standards) and **"No Award"
vendors** (68% of all open cases — even a moderate per-case delay
translates into substantial aggregate business impact given this scale).
Gold-tier vendors, by contrast, process fast enough to plausibly support
early-payment discount terms where such arrangements exist.

**Part 2 — Vendor tier prediction:** Model B estimates a full probability
distribution across all three award tiers for new or thin-history
vendors — not just a single label — letting a user judge how much to
trust each prediction. In practice: **treat predictions with a clear top
probability (>80–90%) as reliable**; treat a close call between adjacent
tiers (e.g., Bronze vs. Silver+, probabilities within ~15 percentage
points) as "likely better than average, but the exact tier is uncertain"
— worth a closer manual look rather than an automatic classification.

Selecting three representative cases, the model reveals the following:

| Case | Item Category | Order Value | Bronze | No Award | Silver+ | Predicted | Actual |
|---|---|---|---|---|---|---|---|
| A | Sales | €2 | 0.1% | 0.0% | **99.9%** | Silver+ | Silver+ ✓ |
| B | Packaging | €18,984 | 0.0% | **100.0%** | 0.0% | No Award | No Award ✓ |
| C | Sales | €31 | **53.6%** | 8.2% | 38.2% | Bronze | Silver+ ✗ |

Cases A and B illustrate a confident, correct prediction (>99% top
probability); Case C illustrates the model's known limitation — a genuine
Silver+ vendor predicted as Bronze, with the top two probabilities only
~15 percentage points apart, exactly the kind of close call the guidance
above warns against treating as definitive.

**Overall**, neither Part 1's nor Part 2's models achieve strong
predictive power in isolation. **Model A (existing vendors) is
particularly limited** — with only 445 rated vendors available, neither
Logistic Regression nor Decision Tree reached strong performance (macro
F1 0.455/0.416), a genuine sample-size constraint rather than a fixable
modeling gap. It is therefore **not recommended for standalone business
use** in its current form. However, the two genuine **champion models** —
**tuned XGBoost** for throughput prediction and **Random Forest** for
Model B's vendor-tier prediction — are trained on a substantially larger
dataset and show promising results; they **could support — not replace —
manual review**, especially for close calls, serving as an early-warning
and monitoring tool to flag which open cases and which vendors need a
closer look.

## Limitations & Further Research

*See Notebook 6, Section 6.6 for the full discussion of each point below.*

**Methodological trade-offs:** A strict as-of-GR prediction point (Part 1)
excludes some potentially predictive signals to avoid hindsight leakage;
vendor-grouped splitting prevents data leakage at the cost of
fine-grained class-balance control; merging Silver and Gold into Silver+
resolved a small-sample problem for Model A, though Model B's larger
scale might support keeping them separate.

**Known limitations:** Silver+ remains difficult to predict across every
model tested — likely a genuine data limitation, not a fixable modeling
gap. Model B's predictions rest heavily on just two features
(`spend_classification_NPR` and `order_value`, 58% combined importance),
so its robustness depends on those two fields' ongoing data quality. Only
four mainstream models were tested per part; LightGBM/CatBoost were
considered but not included given time constraints.

**Further research:** A staged extension predicting from Invoice Receipt
onward (once `gr_to_ir_days` is known) could improve on Part 1's current
as-of-GR model; keeping Gold/Silver separate for Model B specifically;
testing LightGBM/CatBoost for a fuller comparison; investigating whether
the `sub_spend_area = Labels` throughput gap reflects a genuine process
bottleneck or a data artifact. See [Planned Extensions](#planned-extensions)
for the OCPM and social network analysis extensions.

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
