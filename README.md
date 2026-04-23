# datacleaninglabRyan

## Overview
This project builds two data preparation pipelines using different datasets.
The goal is to practice the full data prep workflow from raw CSV to clean,
partitioned datasets ready for modeling.

---

## Datasets

### 1. Job Placement (`Placement_Data_Full_Class.csv`)
MBA student placement records. Contains academic scores (10th grade through MBA),
work experience, specialization, and placement outcome/salary.

### 2. College Completion (`cc_institution_details.csv`)
Institutional-level data on US colleges: graduation rates, retention, SAT scores,
expenditures, endowments, and Carnegie classification characteristics.

---

## Questions Addressed

### Job Placement
**Question:** Can we predict the salary of a placed MBA student based on their
academic background, work experience, and specialization?
- **Type:** Regression
- **Target:** `salary` (continuous; placed students only)
- **Business Metric:** Placement offices can benchmark which student profiles
  command premium offers and direct career coaching resources accordingly.

### College Completion
**Question:** Can we predict whether a college has an above-average graduation rate
based on its institutional characteristics?
- **Type:** Binary classification
- **Target:** `high_grad` — 1 if `grad_150_value` > 41.1% (median), else 0
- **Business Metric:** A state higher-ed board can flag at-risk institutions
  before graduation rates decline, directing intervention funding proactively.

---

## Repository Structure
.
├── README.md            # Assignment details and context
├── pipeline_q1_q3.py    # Questions 1–3: exploration, prep steps, both datasets
└── pipeline_q4.py       # Question 4: reusable pipeline functions (DAG)


---

## Data Prep Steps Applied

1. Correct variable types
2. Collapse rare factor levels
3. One-hot encode categorical variables
4. Normalize continuous variables (Min-Max scaling)
5. Drop unneeded variables
6. Create target variable
7. Calculate target prevalence
8. Train / Tune / Test split (60 / 20 / 20)

---

## Instincts & Concerns

### Job Placement
- **Strengths:** Very clean data, unambiguous target, zero missing salary values among placed students.
- **Concerns:** Only 148 usable rows after filtering to placed students — very small for regression. Salary is right-skewed with a few extreme outliers. Academic score columns are likely correlated (multicollinearity). Data appears to come from a single Indian institution, limiting generalizability.

### College Completion
- **Strengths:** Large dataset (3,798 rows), rich features, balanced target (~46% positive).
- **Concerns:** ~92% of VSA columns are missing and must be dropped. SAT scores missing for ~65% of rows (community colleges don't report). `hbcu` and `flagship` are indicator columns (X vs NaN) requiring conversion. `basic` (Carnegie classification) has 30+ levels requiring collapsing.

---

## How to Run

```bash
pip install pandas numpy scikit-learn
python pipeline_q1_q3.py
python pipeline_q4.py
```
