## Project Background & Motivation

This project did **not** start as a classical machine-learning task.

The initial goal was to assess whether a **pure LLM-based escalation classifier** could reliably decide if a clinical report requires escalation.  
Early experiments quickly revealed fundamental limitations.

---

## 🟢 **Phase 1** – Insight Consolidation (Completed)

### Status
**Completed**

### Key Observations
- A pure LLM-based approach tends to **over-escalate**
- Most post-processing strategies were quickly exhausted
- A cleaner separation between false positives and false negatives proved **structurally impossible**
- Improving recall inevitably increased false positives

### Core Insight
> **The LLM is not a decision-maker.**  
> It is best used as a **structured feature generator**.

### Outcome
- Escalation decisions should **not** be made directly by the LLM
- Instead, the LLM output is transformed into:
  - structured signals
  - confidence estimates
  - categorical assessments
- These signals are then evaluated by a **transparent, controllable ML model**

This insight marks the transition from *LLM-only classification* to a **hybrid LLM + classical ML architecture**.

---

## 🟡 **Phase 2:** LLM Escalation Detection – ML models
The following sections describe the implementation and evaluation of the first classical ML baseline (Logistic Regression) built on top of LLM-derived features.

This phase covers creating a suitable **Logistic Regression model** and **for classifying *Reports pre-evaluated by LLM*.   
The model's purpose is to predict whether a clinical report requires escalation or not, using **structured features derived from LLM outputs**.

This phase covers three goals:
1. Provide a **transparent, explainable reference model**
2. Act as a **sanity check** against LLM-based approaches
3. Establish a **robust evaluation framework** (group-aware CV, threshold optimization)
4. Optmize the model's perfomrance by tunging its hyperparameters

---

### Current Scope

- Binary classification: `escalation` vs. `no_escalation`
- Models: **Logistic Regression**
- Features:
  - severity
  - uncertainty_level
  - confidence
  - clarity
  - domain
  - n_risk_factors
  - n_missing_information
- Preprocessing:
  - One-Hot Encoding (categorical)
  - Standard Scaling (numerical)
- Evaluation focus:
  - **F2-score (primary)**
  - Precision / Recall
  - ROC-AUC, PR-AUC (secondary)

---

## Evaluation Strategy

Two complementary group-aware validation schemes are used:

### 1. GroupShuffleSplit (Monte-Carlo CV)
- Purpose: **stability & variance analysis**
- Repeated random splits on group level
- Used for:
  - Threshold sweep
  - Robust threshold selection
  - Metric distribution analysis (quantiles, IQR)

### 2. GroupKFold (Deterministic CV)
- Purpose: **stress testing & worst-case analysis**
- Each group appears exactly once in the test set
- Used to:
  - Identify fragile folds
  - Expose sensitivity to group composition

---

## Threshold Optimization

- Thresholds are **not fixed at 0.5**
- A **threshold sweep** is performed per split
- Optimal threshold selected based on **F2-score**
- Final threshold intended to be derived from:
  - Median or lower quantile (e.g. q25) of shuffle-based CV

---

## Project Structure (simplified)
```
src/
├── A_report_escalation.py
├── B1_rule_escalation.py
├── B2_llm_escalation.py
├── C1_llm_postprocess.py
├── C2_train_logreg.py          # ML Training entry point
├── D1_evaluation.py
├── D2_single_run.py        
├── D3_group_split-cv.py        # Group-aware cross-validation
├── utils/
│ ├── decision_helper.py
│ ├── escalation_helper.py
│ ├── evaluation_helper.py
│ ├── file_helper.py
│ ├── general_helper.py
│ ├── llm_helper.py
│ ├── mlflow_helper.py
│ ├── path_helper.py
│ ├── preprocess_helper.py
│ └── visualisation_helper.py
├── core/
│ ├──  logger.py
│ ├──  mlflow_logger.py
│ └──  session.py
├── configuration/
│ ├──  A_llm_baseline.py
│ ├──  A_rule_based.py
│ ├──  B1_llm_post.py
│ ├──  B2_llm_post.py
│ ├──  B3_llm_post.py
│ ├──  B4_llm_post.py
│ ├──  C1_logreg_base_single.py
│ ├──  C2_logreg_base_group_shuffle.py
│ └──  C3_logreg_base_group_kfold.py
```
---

## Version History

### v0.4.0 – Postprocessing final
- 

### v0.5.0 Adding LogReg baseline model
**Initial ML Baseline**
- Logistic Regression with OHE + scaling
- Train/test split
- Basic evaluation metrics

**Group-Aware Evaluation**
- GroupShuffleSplit introduced
- GroupKFold cross-validation added
- Detection of strong fold-dependent variance

**Threshold Optimization & Robust CV**
- Threshold sweep analysis (F2-based)
- Separation of:
  - shuffle-based stability analysis
  - kfold-based stress testing
- Quantile-based CV reporting
- Local + MLflow-compatible result logging

---

## Notes

- High metric values on individual splits **do not imply general robustness**
- Worst-case folds are considered **operationally relevant**
- This ML baseline is intentionally simple to maximize interpretability

<!-- ---

## Next Planned Steps

- Fix global decision threshold based on shuffle-CV
- Final GroupKFold evaluation with fixed threshold
- Comparison against LLM-based escalation classifier -->
