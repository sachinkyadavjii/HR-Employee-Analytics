"""
data_generator.py
------------------
Generates a realistic SYNTHETIC HR employee dataset for analytics and
attrition-prediction practice.

The data is randomly generated, but attrition is NOT purely random -
it is nudged by realistic HR risk factors (overtime, low satisfaction,
low pay, long commute, no recent promotion, business travel, etc.)
so that the downstream EDA / SQL / ML work has real patterns to find.

Run directly to (re)create data/hr_employee_data.csv:
    python src/data_generator.py
"""

import numpy as np
import pandas as pd
import os

# Fixed seed so the dataset is reproducible for anyone who clones the repo
np.random.seed(42)

N_EMPLOYEES = 1200  # >1000 records as required

DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]

JOB_ROLES_BY_DEPT = {
    "Sales": ["Sales Executive", "Sales Representative", "Sales Manager"],
    "Research & Development": [
        "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Research Director", "Healthcare Representative",
    ],
    "Human Resources": ["Human Resources", "HR Manager", "Recruiter"],
}

EDUCATION_FIELDS = [
    "Life Sciences", "Medical", "Marketing", "Technical Degree",
    "Human Resources", "Other",
]

MARITAL_STATUS = ["Single", "Married", "Divorced"]
BUSINESS_TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
EMPLOYMENT_TYPE = ["Full-Time", "Part-Time", "Contract"]
GENDER = ["Male", "Female"]

# Probability weights so the company "shape" looks like a real org
DEPT_WEIGHTS = [0.32, 0.55, 0.13]
TRAVEL_WEIGHTS = [0.10, 0.70, 0.20]
EMPLOYMENT_WEIGHTS = [0.82, 0.10, 0.08]


def _pick_job_role(department: str) -> str:
    return np.random.choice(JOB_ROLES_BY_DEPT[department])


def generate_dataset(n: int = N_EMPLOYEES) -> pd.DataFrame:
    """Builds the full synthetic employee dataframe."""

    df = pd.DataFrame()
    df["EmployeeID"] = [f"EMP{str(i).zfill(5)}" for i in range(1, n + 1)]

    df["Age"] = np.random.randint(20, 60, size=n)
    df["Gender"] = np.random.choice(GENDER, size=n, p=[0.6, 0.4])
    df["Department"] = np.random.choice(DEPARTMENTS, size=n, p=DEPT_WEIGHTS)
    df["JobRole"] = df["Department"].apply(_pick_job_role)
    df["Education"] = np.random.choice([1, 2, 3, 4, 5], size=n,
                                        p=[0.10, 0.20, 0.35, 0.25, 0.10])
    df["EducationField"] = np.random.choice(EDUCATION_FIELDS, size=n)
    df["MaritalStatus"] = np.random.choice(MARITAL_STATUS, size=n,
                                            p=[0.35, 0.50, 0.15])

    # Job level correlates loosely with age/experience
    df["JobLevel"] = np.clip(
        (df["Age"] - 20) // 8 + np.random.randint(1, 3, size=n), 1, 5
    )

    df["TotalWorkingYears"] = np.clip(
        df["Age"] - 20 - np.random.randint(0, 4, size=n), 0, None
    )
    df["YearsAtCompany"] = np.clip(
        (df["TotalWorkingYears"] * np.random.uniform(0.2, 1.0, size=n)).astype(int),
        0, df["TotalWorkingYears"]
    )
    df["YearsInCurrentRole"] = np.clip(
        (df["YearsAtCompany"] * np.random.uniform(0.2, 1.0, size=n)).astype(int),
        0, df["YearsAtCompany"]
    )
    df["YearsSinceLastPromotion"] = np.clip(
        (df["YearsAtCompany"] * np.random.uniform(0.0, 0.8, size=n)).astype(int),
        0, df["YearsAtCompany"]
    )

    # Monthly income depends on job level + a bit of randomness (in INR)
    base_income = 18000 + df["JobLevel"] * 15000
    df["MonthlyIncome"] = (base_income * np.random.uniform(0.85, 1.35, size=n)).astype(int)
    df["HourlyRate"] = np.random.randint(30, 100, size=n)

    df["JobSatisfaction"] = np.random.choice([1, 2, 3, 4], size=n,
                                              p=[0.15, 0.20, 0.35, 0.30])
    df["EnvironmentSatisfaction"] = np.random.choice([1, 2, 3, 4], size=n,
                                                      p=[0.15, 0.20, 0.35, 0.30])
    df["RelationshipSatisfaction"] = np.random.choice([1, 2, 3, 4], size=n,
                                                       p=[0.15, 0.20, 0.35, 0.30])
    df["WorkLifeBalance"] = np.random.choice([1, 2, 3, 4], size=n,
                                              p=[0.10, 0.25, 0.45, 0.20])

    df["OverTime"] = np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    df["BusinessTravel"] = np.random.choice(BUSINESS_TRAVEL, size=n, p=TRAVEL_WEIGHTS)
    df["PerformanceRating"] = np.random.choice([1, 2, 3, 4], size=n,
                                                p=[0.05, 0.15, 0.60, 0.20])
    df["TrainingTimesLastYear"] = np.random.randint(0, 7, size=n)
    df["NumCompaniesWorked"] = np.random.randint(0, 8, size=n)
    df["DistanceFromHome"] = np.random.randint(1, 40, size=n)
    df["JobInvolvement"] = np.random.choice([1, 2, 3, 4], size=n,
                                             p=[0.10, 0.25, 0.45, 0.20])
    df["StockOptionLevel"] = np.random.choice([0, 1, 2, 3], size=n,
                                               p=[0.45, 0.30, 0.15, 0.10])

    # Whether the employee received a promotion within the last 5 years.
    # Generated independently of YearsSinceLastPromotion (a separate tenure
    # metric) so it doesn't trivially reduce to "how new is this employee" -
    # that would create a misleading reversed correlation with attrition.
    df["PromotionLast5Years"] = np.random.choice([0, 1], size=n, p=[0.35, 0.65])

    df["EmploymentType"] = np.random.choice(EMPLOYMENT_TYPE, size=n, p=EMPLOYMENT_WEIGHTS)

    # ------------------------------------------------------------------
    # Build an "attrition risk score" from realistic HR factors, then
    # convert it into a probability so attrition is data-driven but not
    # perfectly deterministic (real life is noisy).
    # ------------------------------------------------------------------
    risk = np.zeros(n)

    risk += np.where(df["OverTime"] == "Yes", 1.4, 0)
    risk += (4 - df["JobSatisfaction"]) * 0.45
    risk += (4 - df["WorkLifeBalance"]) * 0.40
    risk += (4 - df["EnvironmentSatisfaction"]) * 0.20

    # Low income relative to job level increases risk
    income_percentile = df["MonthlyIncome"].rank(pct=True)
    risk += (1 - income_percentile) * 1.3

    # Newer employees are more of a flight risk
    risk += np.where(df["YearsAtCompany"] <= 2, 1.0, 0)
    risk += np.where(df["YearsAtCompany"] >= 10, -0.6, 0)

    risk += (df["DistanceFromHome"] / 40) * 0.8
    risk += np.where(df["YearsSinceLastPromotion"] >= 4, 0.5, 0)
    risk += np.where(df["PromotionLast5Years"] == 0, 0.6, 0)
    risk += np.where(df["BusinessTravel"] == "Travel_Frequently", 0.6, 0)
    risk += np.where(df["BusinessTravel"] == "Non-Travel", -0.3, 0)
    risk += (5 - df["JobLevel"]) * 0.15
    risk += np.where(df["MaritalStatus"] == "Single", 0.4, 0)
    risk += np.where(df["StockOptionLevel"] == 0, 0.3, 0)

    # Random noise so relationships are realistic, not perfect
    risk += np.random.normal(0, 1.1, size=n)

    # Squash into a probability and shift so the overall attrition rate
    # lands in a realistic ~15-20% range (typical of real HR datasets)
    # while still preserving the relative risk ordering between employees.
    prob = 1 / (1 + np.exp(-(risk - 5.9) * 0.9))
    df["Attrition"] = np.where(np.random.uniform(0, 1, size=n) < prob, "Yes", "No")

    # Reasonable column order
    column_order = [
        "EmployeeID", "Age", "Gender", "Department", "JobRole", "Education",
        "EducationField", "MaritalStatus", "JobLevel", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "TotalWorkingYears",
        "MonthlyIncome", "HourlyRate", "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance", "OverTime", "BusinessTravel",
        "PerformanceRating", "TrainingTimesLastYear", "NumCompaniesWorked",
        "DistanceFromHome", "JobInvolvement", "StockOptionLevel",
        "PromotionLast5Years", "EmploymentType", "Attrition",
    ]
    return df[column_order]


def main():
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hr_employee_data.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} employee records -> {out_path}")
    print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
