# Invoice-to-Payment Process Optimization & Delay Prediction

Process mining and machine learning applied to a SAP procurement data set of a large multinational company headquartered in the Netherlands containing 1,595,923 events across 251,734 cases (BPI Challenge 2019) — combining PM4Py, scikit-learn, and Power BI to discover processes, analyze throughput, check conformance and predict delays or deviations.

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

## Key Steps

1. Load and clean the BPI 2019 event log (pandas)
2. Process discovery — discover the as-is process with PM4Py (Inductive Miner)
3. Conformance checking — identify deviations from the ideal process flow
4. Process enhancement — bottleneck and delay analysis using timestamp data
5. Feature engineering for delay prediction
6. Train and tune a delay-prediction model (scikit-learn + Optuna), tracked with MLflow
7. Explain model predictions with SHAP
8. Build a Power BI dashboard for business-facing KPIs, including a native Python-visual embedding of PM4Py directly inside Power BI
9. Demonstrate the same process discovery step using an industry-standard commercial tool (Process.Science / Disco) alongside the open-source pipeline
10. (Optional, time-permitting) Performance spectrum analysis using R's psmineR package
11. (Optional, time-permitting) Side-by-side comparison of free Power BI process-mining visuals (Process.Science, Microsoft's native Power Automate Process Mining visual)

## Business Problem

The dataset originates from the [BPI Challenge 2019](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/): 
a large multinational company headquartered in the Netherlands, operating in the coatings and paints industry across 60 subsidiaries, submitted its purchase order handling process for investigation. The process owner's motivation was **compliance** — understanding not just how the process runs on average, but where and how severely it deviates from expectation.

This project follows the three original questions posed by the BPI 
Challenge, extended with a predictive layer:

1. **Process discovery:** is there a collection of process models that 
   together properly describe the process captured in this data? (The 
   challenge itself identifies at least four underlying flow types — 
   3-way matching with GR-based invoicing, 3-way matching without, 2-way 
   matching, and consignment.)
2. **Throughput analysis (enhancement):** what is the throughput of the 
   invoicing process — the time between goods receipt, invoice receipt, 
   and payment (invoice clearing) — including matching the correct goods 
   receipts to invoices when a single line item has several of each?
3. **Conformance and deviation:** which purchase documents stand out from 
   the log, where do they deviate from the discovered process models, and 
   how severe are these deviations — both in terms of process flow and 
   invoice values (e.g., vendors producing disproportionate rework due to 
   invoice errors)?
4. **Prediction (this project's extension):** can invoices likely to 
   experience delay or deviation be flagged early enough for intervention, 
   before payment is late?

## Dataset

- **Source:** BPI Challenge 2019 — SAP procurement event log, published via 
  [4TU.ResearchData](https://data.4tu.nl/articles/_/12715853/1) 
  ([direct file download](https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828), 
  `BPI_Challenge_2019.xes`, ~695 MB uncompressed)
- **License:** CC BY 4.0 (dataset) — separate from this repository's MIT 
  license, which covers code only. Citation: van Dongen, B.F., *BPI 
  Challenge 2019*. 4TU.ResearchData.
- **Format:** IEEE XES standard, read natively via `pm4py.read_xes()`
- **Scope:** 1,595,923 events across 251,734 cases (case ID = purchase 
  document + item), spanning 76,349 purchase documents, 42 activities, and 
  627 users (607 human, 20 batch/automated) — covering purchase orders 
  submitted in 2018 across 60 subsidiaries
- **Key attributes:** case ID, activity, timestamp, resource (user), 
  purchasing document ID, item type, item category (3-way with/without 
  GR-based invoicing, 2-way, consignment), vendor, company (subsidiary), 
  spend classification text, GR-based invoice verification flag, goods 
  receipt flag
- **Note:** the raw `.xes` file is not committed to this repository due to 
  its size — see the link above to download it directly. `data/raw/` is 
  excluded via `.gitignore`.

## Approach

- **Process Mining:** PM4Py for process discovery (Inductive Miner) and conformance checking against the expected purchase-to-pay flow
- **Segmented analysis:** where relevant, process discovery, conformance checking, and duration analysis are performed both in aggregate and segmented by item category (the four flow types), vendor, subsidiary, and time period — since aggregate metrics can mask meaningful variation across these dimensions (e.g., a company-wide average duration can look acceptable while masking poor performance concentrated in a few subsidiaries).
-  **Conformance checking:** BPI 2019 does not include a formal, machine-readable reference (de jure) process model. In practice, de jure models originate from artifacts such as internal process handbooks, audit requirements, or regulatory/legal frameworks — none of which are part of this public research dataset, since such governance documentation is typically internal and confidential. Both winning  BPI Challenge 2019 submissions faced this same gap: [Augusto, Leno 
  & Reissner (2019)](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Student-Submission-1.pdf) constructed as-is/to-be BPMN models but found them too complex for 
  automated conformance techniques; [Diba, Remy & Pufahl (2019)](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Submission-6.pdf) (overall challenge winners) instead applied rule-based compliance checking. Following this precedent, this project checks conformance against two baselines: (1) a de facto model discovered from the log's dominant behavior, and (2) a lightweight de jure reference encoding the four-flow-type description shared by both prior submissions — the same textual specification already used in this project's Business Problem section.
- **Machine Learning:** delay-prediction model (scikit-learn), tuned via Optuna, tracked in MLflow
- **Explainability:** SHAP for global and individual-case feature importance, complemented by a dtreeviz visualization of a representative decision tree for structural interpretability
- **Visualization:** Power BI dashboard for business-facing KPIs, built on model predictions scored offline and exported as a table, including a native
  Python-visual embedding of PM4Py (via [viadee's](https://www.viadee.de/en/blog/process-mining-mit-power-bi/) approach)
- **Industry Tool Demonstration:** limited demos of Process.Science's free Power BI visual and Disco (30-day trial), cross-validating the open-source pipeline against commercial tools on a filtered data subset
- **Stretch goals:** R-based performance spectrum analysis ([psmineR](https://cran.r-project.org/web/packages/psmineR/index.html)); side-by-side comparison of free Power BI process-mining visuals
- **Stretch goal Object-centric process mining:** BPI 2019's case 
  notion (purchase document + item) flattens what is actually a 
  multi-object process (Purchase Order, Item, Goods Receipt, Invoice). 
  A future extension could convert the event log to OCEL 2.0 format 
  and apply PM4Py's object-centric discovery (OC-DFG, OC-Petri nets) 
  to more accurately capture one-to-many relationships that the 
  traditional single-case-notion analysis in this project simplifies.
- **(Stretch goal) Social network analysis:** PM4Py's native 
  organizational mining functions (`discover_handover_of_work_network`, 
  `discover_working_together_network`) applied to the log's 627 
  resources, to explore who handovers concentrate around and whether 
  central resources correspond to process bottlenecks. Network 
  centrality alone shows frequency of connection, not speed — so this 
  is cross-referenced against Step 4's duration/throughput data before 
  drawing any bottleneck conclusions, rather than inferred from the 
  network visualization alone.

  **Note:** TU Eindhoven has published an official graph-based object-centric 
representation of this dataset ([Event Graph of BPI Challenge 2019](https://doi.org/10.4121/14169614), 
Esser & Fahland, 2021), modeling PO, POItem, Resource, and Vendor as 
distinct entities. Consistent with that work, this project's OCEL 
conversion also treats Goods Receipt and Invoice as derived (rather 
than natively identified) entities, since neither the original XES 
export nor TU/e's own graph model include distinct document IDs for 
individual goods receipts or invoices.

## Planned Extensions
These extensions are committed and will be completed — the open 
question is timing, not whether. They are deliberately decoupled from 
the September 20, 2026 deadline so they don't compete with the core 
pipeline under time pressure.

**Social network analysis** *(lower complexity — likely first)*
PM4Py's native handover-of-work and working-together networks, 
analyzing resource collaboration patterns across the 627 users in the 
log, cross-referenced with duration data from Step 4 to distinguish 
genuine bottlenecks from high-throughput specialists.

*Extension: workload vs. service time.* Building on Nakatumba & van der 
Aalst's application of the Yerkes-Dodson Law of Arousal to process 
mining ([BPM 2009 workshop paper](https://doi.org/10.1007/978-3-642-12186-9_8)), 
this analysis will compute each resource's concurrent workload at the 
time of each completed activity and test — via regression — whether 
service time follows the predicted inverted-U pattern: moderate 
workload correlating with faster performance, with degradation at both 
very low and very high workload.

**Object-centric process mining** *(higher complexity — "Project 1 v2")*
Object-centric process mining (OCEL 2.0 conversion + PM4Py's OC-DFG/
OC-Petri net discovery) is planned as a post-launch extension, to more
accurately capture the one-to-many relationships (e.g., multiple goods
receipts and invoices per line item) that this project's core analysis
simplifies via a single case notion. TU Eindhoven's own graph-based
object-centric representation of this dataset ([Esser & Fahland, 2021](https://doi.org/10.4121/14169614))
independently confirms this modeling gap — their object model also
treats only PO, POItem, Resource, and Vendor as distinct entities.

This extension will explore questions that specifically exploit the
object-centric view, building directly on the original BPI Challenge's
own compliance framing:
- **Object multiplicity vs. delay risk:** for purchase order items with
  multiple Goods Receipt and Invoice objects (e.g., 12 GRs/invoices for
  a single rent line item), does the number of related objects
  correlate with total case duration or deviation severity?
- **GR–Invoice desynchronization:** at the object level, how long does
  a specific Invoice object wait for its corresponding Goods Receipt
  object (or vice versa), and does this gap vary systematically by
  vendor or subsidiary?
- **Vendor object-interaction signatures:** can vendors be segmented by
  their characteristic object-interaction pattern (one-to-one vs.
  high-multiplicity GR/Invoice relationships), and do higher-multiplicity
  vendors show more conformance deviations or invoice-value mismatches?

## Key Findings

*(To be completed once analysis is finished.)*

## Tools & Technologies

PM4Py · scikit-learn · Optuna · MLflow · SHAP · dtreeviz · Power BI (incl. native Python visual integration) · Process.Science (Power BI visual) · Disco · psmineR (R, stretch goal) · Power Automate Process Mining visual (stretch goal) · Claude (Anthropic)

## Repository Structure

```
invoice-to-payment-process-optimization/
│
├── README.md
├── LICENSE                          # MIT (code only — see Dataset section for data license)
├── requirements.txt
├── .gitignore
│
├── data/
│   └── BPI Challenge 2019_Data/     # both raw and processed files together
│       ├── BPI_Challenge_2019.xes         # raw file (linked, not committed — see Dataset section)
│       └── BPI_2019_cleaned.parquet       # cleaned output from Notebook 1
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_process_discovery.ipynb        # Inductive Miner, process map
│   ├── 03_conformance_checking.ipynb     # deviations, de facto vs. de jure
│   ├── 04_process_enhancement.ipynb      # bottlenecks, throughput, variants
│   ├── 05_feature_engineering.ipynb      # case duration, delay labels, etc.
│   ├── 06_delay_prediction_model.ipynb   # scikit-learn + Optuna tuning
│   └── 07_shap_explainability.ipynb
│
├── extensions/                      # Planned Extensions (see README section below) —
│   │                                 # committed, but not deadline-bound; not part of
│   │                                 # the numbered core sequence above
│   ├── social_network_analysis.ipynb
│   └── ocpm_extension.ipynb
│
├── powerbi/
│   ├── invoice_dashboard.pbix
│   └── screenshots/
│       └── process_science_discovery.png
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
