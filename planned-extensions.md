# Planned Extensions

These extensions are committed and will be completed — the open question is
timing, not whether. They are deliberately decoupled from the September 20,
2026 deadline so they don't compete with the core pipeline under time
pressure.

*See the [README](../README.md) for the project overview and core findings.
This document holds the detailed methodology, specific research questions,
and citations for each planned extension.*

---

## Social network analysis *(lower complexity — likely first)*

PM4Py's native handover-of-work and working-together networks, analyzing
resource collaboration patterns across the 627 users in the log,
cross-referenced with duration data from Step 4 to distinguish genuine
bottlenecks from high-throughput specialists.

**Extension: workload vs. service time.** Building on Nakatumba & van der
Aalst's application of the Yerkes-Dodson Law of Arousal to process mining
([BPM 2009 workshop paper](https://doi.org/10.1007/978-3-642-12186-9_8)),
this analysis will compute each resource's concurrent workload at the time
of each completed activity and test — via regression — whether service time
follows the predicted inverted-U pattern: moderate workload correlating with
faster performance, with degradation at both very low and very high workload.

---

## Object-centric process mining *(higher complexity — "Project 1 v2")*

Object-centric process mining (OCEL 2.0 conversion + PM4Py's OC-DFG/
OC-Petri net discovery) is planned as a post-launch extension, to more
accurately capture the one-to-many relationships (e.g., multiple goods
receipts and invoices per line item) that this project's core analysis
simplifies via a single case notion. TU Eindhoven's own graph-based
object-centric representation of this dataset
([Esser & Fahland, 2021](https://doi.org/10.4121/14169614)) independently
confirms this modeling gap — their object model also treats only PO,
POItem, Resource, and Vendor as distinct entities.

Notebook 3's conformance checking already surfaced direct empirical
motivation for this extension: high-multiplicity cases deviate from the
de facto model far more than baseline (22.12% vs. 3.84%), yet show *lower*
de jure rule violations — indicating these cases are structurally complex
(multiple GR/invoice objects) rather than genuinely non-compliant, a
distinction a single-case-notion model cannot represent cleanly. This
extension will explore questions that specifically exploit the
object-centric view, building directly on the original BPI Challenge's own
compliance framing:

- **Object multiplicity vs. delay risk:** for purchase order items with
  multiple Goods Receipt and Invoice objects (e.g., 12 GRs/invoices for a
  single rent line item), does the number of related objects correlate
  with total case duration or deviation severity?
- **GR–Invoice desynchronization:** at the object level, how long does a
  specific Invoice object wait for its corresponding Goods Receipt object
  (or vice versa), and does this gap vary systematically by vendor or
  subsidiary?
- **Vendor object-interaction signatures:** can vendors be segmented by
  their characteristic object-interaction pattern (one-to-one vs.
  high-multiplicity GR/Invoice relationships), and do higher-multiplicity
  vendors show more conformance deviations or invoice-value mismatches?

---

## Power BI Dashboard & Process.Science Integration *(stretch)*

### Extended Process.Science demonstration *(low complexity)*

The core Industry Tool Demonstration (Step 9) is scoped to two
Process.Science modules (Variants/Case Analyzer, plus one
filtering/drill-down module), directly connecting to this project's own
validated findings (7,835 process variants, top 5 covering 48.3% of cases).
A fuller replication of
[viadee's Power BI process-mining approach](https://www.viadee.de/en/blog/process-mining-mit-power-bi/)
— including throughput-vs-frequency and per-activity duration visuals — is
planned as a post-deadline enhancement to this demo.

### Planned dashboard structure

Informed by two independent precedents also built on this dataset or a
closely related process-mining Power BI use case:

- **Dashboard 1 — Process Overview**: invoicing-type shares by net worth
  and case count, throughput timeline, document/item type breakdowns
  (extends 4.2). Adds a predicted-vs-actual panel from Notebook 6's
  test-set results, plus a filterable view of currently open cases flagged
  by `predicted_gr_to_clear_days` — surfacing which in-progress orders are
  likely to run long
- **Dashboard 2 — Vendor & Spend**: vendor importance, spend area
  analysis, vendor tier ratings (extends 4.5.2, 5.1). Adds Model A/B's
  predicted Award tiers for thin-history vendors, shown alongside actual
  computed tiers where available
- **Dashboard 3 — Rework & Bottlenecks**: rework activity ratios,
  responsible vendors/users, duration impact (extends 4.3) — purely
  descriptive, no model output feeds this dashboard

This structure mirrors the dashboard layout used by
[Rząd, Wojnecka, Rutkowski & Guliński (2019)](https://icpmconference.org/2019/wp-content/uploads/sites/6/2019/07/BPI-Challenge-Submission-2.pdf),
who built a Power BI application on this same dataset.

### Data model

A single flat, enriched event log (case ID, activity, timestamp, plus
case-level attributes) feeds the process-mining visual, alongside the
existing case-level and vendor-level feature/summary tables already built
in this project for the KPI dashboards. No formal star schema is used —
this follows the approach taken by both
[viadee](https://www.viadee.de/en/blog/process-mining-mit-power-bi/) and
Rząd et al., who each built directly from flat/pivoted event-log tables in
Power Query rather than a dimensional model.
[Process.Science's](https://www.process-science.com/solutions/procurement)
own documented input requirements confirm the same pattern: transaction
data enriched with case-level metadata (vendor, category, spend area), not
a dimensional schema.

### Predictive analytics integration (Notebook 6)

Model outputs are exported as simple additions to the existing tables, not
new architecture:

- Part 1's scored predictions for unfinished cases
  (`predicted_gr_to_clear_days`), joined onto the case-level table
- Part 1's model performance summary (RMSE/MAE/R² per model), as a small
  reference table supporting a "how trustworthy is this estimate" panel
- Part 2 Model A's predicted tiers for the 1,222 "Insufficient Data"
  vendors, joined onto the vendor-level table
- Part 2 Model B's predicted tiers per transaction (order-level attributes
  only), for new or thin-history vendors

### SHAP explainability (Notebook 7)

Global and individual-case feature importance from the champion models is
a natural further extension — a "why was this prediction made" panel
alongside the predictions above, now that Notebook 7 is complete.

### Process.Science integration

The free Power BI visual (30-day trial) requires only Case ID, Activity,
and Timestamp — compatible directly with the project's existing
`event_log_p2p` structure. Notably, Process.Science's own use-case
documentation independently names "maverick buying — invoices before
orders are placed" as a detected pattern, corroborating the framing
already cited from van Dyk, Kennes, Aklecha & Ramezani (2019) in
Notebooks 3 and 4.
