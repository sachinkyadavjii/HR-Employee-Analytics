"""
data_cleaning.py
------------------
Loads the raw HR dataset and runs a simple, transparent data-quality
and cleaning pipeline:

  1. Load CSV
  2. Inspect shape / dtypes
  3. Check missing values
  4. Check & remove duplicate rows
  5. Handle missing values
  6. Fix/validate data types
  7. Validate numeric ranges
  8. Validate the Attrition column
  9. Save a cleaned copy + print a data-quality summary

Run directly:
    python src/data_cleaning.py
"""

import os
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hr_employee_data.csv")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hr_employee_data_clean.csv")

NUMERIC_COLUMNS = [
    "Age", "Education", "JobLevel", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "TotalWorkingYears", "MonthlyIncome", "HourlyRate",
    "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
    "WorkLifeBalance", "PerformanceRating", "TrainingTimesLastYear",
    "NumCompaniesWorked", "DistanceFromHome", "JobInvolvement",
    "StockOptionLevel", "PromotionLast5Years",
]

CATEGORICAL_COLUMNS = [
    "Gender", "Department", "JobRole", "EducationField", "MaritalStatus",
    "OverTime", "BusinessTravel", "EmploymentType", "Attrition",
]


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Loads the raw CSV file."""
    df = pd.read_csv(path)
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Returns a series of missing-value counts per column (only columns with > 0)."""
    missing = df.isnull().sum()
    return missing[missing > 0]


def check_duplicates(df: pd.DataFrame) -> int:
    """Returns the number of fully duplicated rows."""
    return int(df.duplicated().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full cleaning pipeline and returns a cleaned dataframe."""

    df = df.copy()

    # 1. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed_dupes = before - len(df)

    # 2. Drop duplicate EmployeeIDs, keeping the first occurrence
    if "EmployeeID" in df.columns:
        df = df.drop_duplicates(subset="EmployeeID", keep="first")

    # 3. Handle missing values
    #    - Numeric columns: fill with median (robust to outliers)
    #    - Categorical columns: fill with mode (most frequent value)
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Convert categorical columns to the pandas 'category' dtype
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 5. Ensure numeric columns are actually numeric
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Validate numeric ranges (clip impossible values instead of dropping rows)
    if "Age" in df.columns:
        df = df[(df["Age"] >= 18) & (df["Age"] <= 70)]
    if "MonthlyIncome" in df.columns:
        df = df[df["MonthlyIncome"] > 0]
    if "DistanceFromHome" in df.columns:
        df["DistanceFromHome"] = df["DistanceFromHome"].clip(lower=0)

    # 7. Validate the Attrition column only contains Yes/No
    if "Attrition" in df.columns:
        df = df[df["Attrition"].isin(["Yes", "No"])]

    # 8. Drop rows with missing critical data (target variable or ID)
    critical_cols = [c for c in ["EmployeeID", "Attrition"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    return df, removed_dupes


def print_quality_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame, removed_dupes: int):
    """Prints a short, readable data-quality summary."""

    print("=" * 60)
    print("HR DATA - DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"Raw rows                : {len(raw_df):,}")
    print(f"Raw columns              : {raw_df.shape[1]}")
    print(f"Duplicate rows removed   : {removed_dupes}")
    print(f"Rows after cleaning      : {len(clean_df):,}")
    print(f"Columns after cleaning   : {clean_df.shape[1]}")

    missing = check_missing_values(raw_df)
    if missing.empty:
        print("Missing values (raw)     : none found")
    else:
        print("Missing values (raw)     :")
        for col, count in missing.items():
            print(f"    - {col}: {count}")

    print(f"Attrition rate (cleaned) : {(clean_df['Attrition'] == 'Yes').mean() * 100:.1f}%")
    print("=" * 60)


def main():
    df = load_data()
    clean_df, removed_dupes = clean_data(df)
    print_quality_summary(df, clean_df, removed_dupes)

    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset saved to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
