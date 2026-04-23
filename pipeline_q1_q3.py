import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# DATASET 1: JOB PLACEMENT
# =============================================================================

# --- Step 1: Question ---
# Can we predict the salary of a placed MBA student based on their academic
# background, work experience, and specialization?

# --- Step 2: Business Metric ---
# Which student profiles command premium salaries?
# The independent business metric is Mean Absolute Error (MAE)

# --- Step 3: Instincts & Concerns ---
# The dataset is clean and the target is unambiguous, salary exists for every
# placed student with no missing values. However, there are only 148 usable rows
# after filtering to placed students, which is very small for a regression problem
# and will likely cause overfitting.

# --- Data Preparation ---

jp = pd.read_csv("Placement_Data_Full_Class.csv")

# Filter to placed students only, salary (target) only exists for placed students
jp = jp[jp["status"] == "Placed"].copy()

# Drop unneeded variables: sl_no is a row ID, status is now constant
jp.drop(columns=["sl_no", "status"], inplace=True)

# Correct variable types, all categoricals are already string, numerics are float
# No type corrections needed

# Collapse factor levels — hsc_s 'Arts' has <6% of placed rows, merge into 'Other'
jp["hsc_s"] = jp["hsc_s"].apply(lambda x: x if x in ["Commerce", "Science"] else "Other")

# One-hot encode categorical variables
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
jp = pd.get_dummies(jp, columns=cat_cols, drop_first=True)

# Normalize continuous variables
num_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
jp[num_cols] = MinMaxScaler().fit_transform(jp[num_cols])

# Target variable: salary is already continuous, no transformation needed
# Prevalence 
print("=== JOB PLACEMENT: Target (salary) Distribution ===")
print(jp["salary"].describe().round(2))

# Create Train / Tune / Test partitions (60 / 20 / 20)
X_jp = jp.drop(columns=["salary"])
y_jp = jp["salary"]
X_train, X_temp, y_train, y_temp = train_test_split(X_jp, y_jp, test_size=0.40, random_state=42)
X_tune, X_test, y_tune, y_test   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"\nSplits — Train: {len(X_train)}, Tune: {len(X_tune)}, Test: {len(X_test)}")


# =============================================================================
# DATASET 2: COLLEGE COMPLETION
# =============================================================================

# --- Step 1: Question ---
# Can we predict whether a college has an above-average graduation rate based
# on its institutional characteristics?

# --- Step 2: Business Metric ---
# The independent business metric is Precision among colleges
# predicted to have low graduation rates 

# --- Step 3: Instincts & Concerns ---
# The dataset is large (3,798 institutions) and the target is well-balanced (~46%
# positive), which is promising. However, the VSA columns are 92%+ missing and
# must be dropped entirely. SAT scores are missing for ~65% of institutions (many
# community colleges don't report them), so that column must also be dropped.

# --- Data Preparation ---

cc = pd.read_csv("cc_institution_details__1_.csv")

# Drop unneeded variables:
# - VSA columns: 92%+ missing
# - Percentile columns: derived from the value columns, redundant
# - ID/metadata columns: no predictive value
# - State-level aggregate columns: not institutional-level predictors
# - grad_100_value: alternative form of the target, would leak
# - med_sat_value: 65%+ missing, too many to impute reliably
drop_cols = (
    [c for c in cc.columns if c.startswith("vsa_") or c.endswith("_percentile")]
    + ["index", "unitid", "chronname", "city", "site", "similar", "nicknames",
       "state", "state_sector_ct", "carnegie_ct", "counted_pct", "long_x", "lat_y",
       "cohort_size", "grad_100_value", "med_sat_value",
       "awards_per_state_value", "awards_per_natl_value",
       "exp_award_state_value", "exp_award_natl_value"]
)
cc.drop(columns=[c for c in drop_cols if c in cc.columns], inplace=True)

# Drop rows where target source column is missing, then create target variable
# Target: 1 if grad_150_value > 41.1 (dataset median), else 0
cc.dropna(subset=["grad_150_value"], inplace=True)
cc["high_grad"] = (cc["grad_150_value"] > 41.1).astype(int)
cc.drop(columns=["grad_150_value"], inplace=True)

# Correct variable types — hbcu and flagship are 'X'/NaN indicators, convert to 0/1
for col in ["hbcu", "flagship"]:
    cc[col] = cc[col].apply(lambda x: 1 if x == "X" else 0)

# Impute missing values with column median (small missingness remaining)
for col in ["retain_value", "aid_value", "pell_value", "ft_pct", "ft_fac_value", "endow_value"]:
    cc[col] = cc[col].fillna(cc[col].median())

# Collapse factor levels — 'basic' has 30+ Carnegie classification levels
def collapse_carnegie(val):
    val = str(val)
    for label in ["Research", "Masters", "Baccalaureate"]:
        if label in val: return label
    return "Associates" if "Associate" in val else "Other"

cc["basic"] = cc["basic"].apply(collapse_carnegie)

# One-hot encode categorical variables
cc = pd.get_dummies(cc, columns=["level", "control", "basic"], drop_first=True)

# Normalize continuous variables
num_cols_cc = [c for c in ["student_count", "awards_per_value", "exp_award_value",
                            "ft_pct", "fte_value", "aid_value", "endow_value",
                            "pell_value", "retain_value", "ft_fac_value"] if c in cc.columns]
cc[num_cols_cc] = MinMaxScaler().fit_transform(cc[num_cols_cc])

# Prevalence of target variable
prevalence = cc["high_grad"].mean()
print(f"\n=== COLLEGE COMPLETION: Target Prevalence ===")
print(f"High graduation rate (1): {cc['high_grad'].sum()} ({prevalence:.1%})")
print(f"Low graduation rate  (0): {(cc['high_grad'] == 0).sum()} ({1 - prevalence:.1%})")

# Create Train / Tune / Test partitions (60 / 20 / 20), stratified to preserve balance
X_cc = cc.drop(columns=["high_grad"])
y_cc = cc["high_grad"]
X_train, X_temp, y_train, y_temp = train_test_split(X_cc, y_cc, test_size=0.40, random_state=42, stratify=y_cc)
X_tune, X_test, y_tune, y_test   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nSplits — Train: {len(X_train)}, Tune: {len(X_tune)}, Test: {len(X_test)}")
