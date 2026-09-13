"""
analysis.py
------------
Shared, reusable analytics functions used by both the Jupyter notebook
(EDA) and the Streamlit dashboard. Keeping this logic in one place means
the dashboard and the notebook always agree on how KPIs are calculated.

All functions take a (possibly filtered) pandas DataFrame and return
either a scalar KPI or a small summary DataFrame - nothing is hardcoded.
"""

import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> dict:
    """Returns a dictionary of top-level HR KPIs calculated from df."""
    total_employees = len(df)
    employees_left = int((df["Attrition"] == "Yes").sum())
    attrition_rate = (employees_left / total_employees * 100) if total_employees else 0

    return {
        "total_employees": total_employees,
        "employees_left": employees_left,
        "attrition_rate": round(attrition_rate, 1),
        "avg_age": round(df["Age"].mean(), 1) if total_employees else 0,
        "avg_monthly_income": round(df["MonthlyIncome"].mean(), 0) if total_employees else 0,
        "avg_experience": round(df["TotalWorkingYears"].mean(), 1) if total_employees else 0,
        "avg_job_satisfaction": round(df["JobSatisfaction"].mean(), 2) if total_employees else 0,
        "avg_training_hours": round(df["TrainingTimesLastYear"].mean(), 1) if total_employees else 0,
    }


def attrition_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Generic helper: attrition count/rate grouped by any categorical column."""
    grouped = df.groupby(column, observed=True)["Attrition"].agg(
        Total="count",
        Left=lambda s: (s == "Yes").sum(),
    )
    grouped["AttritionRate"] = (grouped["Left"] / grouped["Total"] * 100).round(1)
    return grouped.reset_index().sort_values("AttritionRate", ascending=False)


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an AgeGroup column (bins) to the dataframe."""
    df = df.copy()
    bins = [17, 25, 35, 45, 55, 70]
    labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


def add_salary_range(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a SalaryRange column (bins) to the dataframe."""
    df = df.copy()
    bins = [0, 30000, 50000, 70000, 100000, 1_000_000]
    labels = ["<30k", "30k-50k", "50k-70k", "70k-100k", "100k+"]
    df["SalaryRange"] = pd.cut(df["MonthlyIncome"], bins=bins, labels=labels)
    return df


def add_experience_group(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a YearsAtCompanyGroup column (bins) to the dataframe."""
    df = df.copy()
    bins = [-1, 1, 3, 6, 10, 100]
    labels = ["0-1", "2-3", "4-6", "7-10", "10+"]
    df["YearsAtCompanyGroup"] = pd.cut(df["YearsAtCompany"], bins=bins, labels=labels)
    return df


def attrition_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a JobRole x Department pivot table of attrition rate (%)."""
    pivot = df.pivot_table(
        index="JobRole",
        columns="Department",
        values="Attrition",
        aggfunc=lambda s: round((s == "Yes").mean() * 100, 1),
        observed=True,
    )
    return pivot


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns numeric-feature correlation with a binary attrition flag."""
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    corr = numeric_df.corr(numeric_only=True)["AttritionFlag"].drop("AttritionFlag")
    return corr.sort_values(ascending=False)
