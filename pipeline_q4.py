import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def split_data(df, target, stratify=False, random_state=42):
    X, y = df.drop(columns=[target]), df[target]
    strat = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state, stratify=strat)
    X_tune,  X_test, y_tune,  y_test  = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp if stratify else None)
    return (pd.concat([X_train, y_train], axis=1),
            pd.concat([X_tune,  y_tune],  axis=1),
            pd.concat([X_test,  y_test],  axis=1))


# =============================================================================
# JOB PLACEMENT PIPELINE
# =============================================================================

def load_job_placement(filepath):
    return pd.read_csv(filepath)

def filter_placed(df):
    return df[df["status"] == "Placed"].copy()

def drop_jp_columns(df):
    return df.drop(columns=["sl_no", "status"])

def collapse_jp_factors(df):
    df = df.copy()
    df["hsc_s"] = df["hsc_s"].apply(lambda x: x if x in ["Commerce", "Science"] else "Other")
    return df

def encode_jp_categoricals(df):
    cats = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
    return pd.get_dummies(df, columns=cats, drop_first=True)

def normalize_jp_numerics(df):
    df = df.copy()
    cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    df[cols] = MinMaxScaler().fit_transform(df[cols])
    return df

def run_job_placement_pipeline(filepath):
    df = load_job_placement(filepath)
    df = filter_placed(df)
    df = drop_jp_columns(df)
    df = collapse_jp_factors(df)
    df = encode_jp_categoricals(df)
    df = normalize_jp_numerics(df)
    return split_data(df, target="salary")


# =============================================================================
# COLLEGE COMPLETION PIPELINE
# =============================================================================

def load_college_completion(filepath):
    return pd.read_csv(filepath)

def drop_cc_columns(df):
    drop = (
        [c for c in df.columns if c.startswith("vsa_") or c.endswith("_percentile")]
        + ["index", "unitid", "chronname", "city", "site", "similar", "nicknames",
           "state", "state_sector_ct", "carnegie_ct", "counted_pct", "long_x", "lat_y",
           "cohort_size", "grad_100_value", "med_sat_value",
           "awards_per_state_value", "awards_per_natl_value",
           "exp_award_state_value", "exp_award_natl_value"]
    )
    return df.drop(columns=[c for c in drop if c in df.columns])

def create_grad_target(df, median=41.1):
    df = df.dropna(subset=["grad_150_value"]).copy()
    df["high_grad"] = (df["grad_150_value"] > median).astype(int)
    return df.drop(columns=["grad_150_value"])

def fix_cc_indicators(df):
    df = df.copy()
    for col in ["hbcu", "flagship"]:
        df[col] = df[col].apply(lambda x: 1 if x == "X" else 0)
    return df

def impute_cc_missing(df):
    df = df.copy()
    for col in ["retain_value", "aid_value", "pell_value", "ft_pct", "ft_fac_value", "endow_value"]:
        df[col] = df[col].fillna(df[col].median())
    return df

def collapse_carnegie_levels(df):
    def _collapse(val):
        val = str(val)
        for label in ["Research", "Masters", "Baccalaureate"]:
            if label in val: return label
        return "Associates" if "Associate" in val else "Other"
    df = df.copy()
    df["basic"] = df["basic"].apply(_collapse)
    return df

def encode_cc_categoricals(df):
    return pd.get_dummies(df, columns=["level", "control", "basic"], drop_first=True)

def normalize_cc_numerics(df):
    df = df.copy()
    cols = [c for c in ["student_count", "awards_per_value", "exp_award_value",
                         "ft_pct", "fte_value", "aid_value", "endow_value",
                         "pell_value", "retain_value", "ft_fac_value"] if c in df.columns]
    df[cols] = MinMaxScaler().fit_transform(df[cols])
    return df

def run_college_completion_pipeline(filepath):
    df = load_college_completion(filepath)
    df = drop_cc_columns(df)
    df = create_grad_target(df)
    df = fix_cc_indicators(df)
    df = impute_cc_missing(df)
    df = collapse_carnegie_levels(df)
    df = encode_cc_categoricals(df)
    df = normalize_cc_numerics(df)
    return split_data(df, target="high_grad", stratify=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    jp_train, jp_tune, jp_test = run_job_placement_pipeline("Placement_Data_Full_Class.csv")
    cc_train, cc_tune, cc_test = run_college_completion_pipeline("cc_institution_details__1_.csv")

    print(f"JP — Train: {jp_train.shape}, Tune: {jp_tune.shape}, Test: {jp_test.shape}")
    print(f"CC — Train: {cc_train.shape}, Tune: {cc_tune.shape}, Test: {cc_test.shape}")
