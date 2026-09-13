"""
dashboard/app.py
------------------
HR Employee Analytics & Attrition Prediction Dashboard (Streamlit).

Run with:
    streamlit run dashboard/app.py

The dashboard is organised into pages (Overview, Employee Analysis,
Attrition Analysis, Prediction, SQL Insights, About) selected from the
sidebar. All KPIs and charts are calculated live from the dataset -
nothing is hardcoded.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Make sure we can import from src/ regardless of where streamlit is launched from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from analysis import (  # noqa: E402
    calculate_kpis, attrition_rate_by, add_age_group, add_salary_range,
    add_experience_group, attrition_heatmap_data,
)
from prediction import predict_single_employee, MODEL_PATH  # noqa: E402
from database import run_query, DB_PATH  # noqa: E402

DATA_PATH_CLEAN = os.path.join(PROJECT_ROOT, "data", "hr_employee_data_clean.csv")
DATA_PATH_RAW = os.path.join(PROJECT_ROOT, "data", "hr_employee_data.csv")

# ---------------------------------------------------------------------------
# Page config + design tokens (custom CSS)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Employee Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached) - defined early so filter options are available
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    path = DATA_PATH_CLEAN if os.path.exists(DATA_PATH_CLEAN) else DATA_PATH_RAW
    df = pd.read_csv(path)
    df = add_age_group(df)
    df = add_salary_range(df)
    df = add_experience_group(df)
    return df


@st.cache_resource
def model_available() -> bool:
    return os.path.exists(MODEL_PATH)


df_full = load_data()


# ---------------------------------------------------------------------------
# Sidebar - navigation, theme toggle, and filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## HR Analytics")
    st.caption("Employee & attrition intelligence")

    dark_mode = st.toggle("Dark Mode", value=st.session_state.get("dark_mode", False), key="dark_mode")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Employee Analysis", "Attrition Analysis", "Prediction",
         "SQL Insights", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")

    def _options(col):
        return sorted(df_full[col].dropna().unique().tolist())

    with st.form("filter_form"):
        f_department = st.multiselect("Department", _options("Department"))
        f_job_role = st.multiselect("Job Role", _options("JobRole"))
        f_gender = st.multiselect("Gender", _options("Gender"))
        f_employment_type = st.multiselect("Employment Type", _options("EmploymentType"))
        f_age_group = st.multiselect("Age Group", _options("AgeGroup"))
        f_overtime = st.multiselect("OverTime", _options("OverTime"))

        col_a, col_b = st.columns(2)
        apply_clicked = col_a.form_submit_button("Apply Filters", width='stretch')
        reset_clicked = col_b.form_submit_button("Reset Filters", width='stretch')

    if reset_clicked:
        st.session_state["filters"] = {}
        st.rerun()

    if apply_clicked or "filters" not in st.session_state:
        st.session_state["filters"] = {
            "Department": f_department,
            "JobRole": f_job_role,
            "Gender": f_gender,
            "EmploymentType": f_employment_type,
            "AgeGroup": f_age_group,
            "OverTime": f_overtime,
        }


# ---------------------------------------------------------------------------
# Design tokens (depend on the Dark Mode toggle above) + custom CSS
# ---------------------------------------------------------------------------
PRIMARY_DARK = "#152238"      # sidebar background (always dark, both themes)
ACCENT_TEAL = "#1D9E75"       # positive / brand accent
ACCENT_AMBER = "#BA7517"      # medium risk
ACCENT_RED = "#A32D2D"        # high risk / attrition
TEXT_LIGHT = "#E8ECF1"        # sidebar text (always light, both themes)

if dark_mode:
    BG_MAIN = "#10151F"
    CARD_BG = "#1B222E"
    BORDER_COLOR = "#2A3240"
    TEXT_MAIN = "#E8ECF1"
    TEXT_MUTED = "#9AA4B2"
    TEXT_SUBTLE = "#7C8695"
else:
    BG_MAIN = "#F5F6F8"
    CARD_BG = "#FFFFFF"
    BORDER_COLOR = "#E3E6EB"
    TEXT_MAIN = "#101828"
    TEXT_MUTED = "#6B7280"
    TEXT_SUBTLE = "#667085"

# Plotly charts follow the same toggle automatically (background, gridlines,
# fonts) without needing to touch every chart-building line below.
pio.templates.default = "plotly_dark" if dark_mode else "plotly_white"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG_MAIN};
}}

/* Sidebar - stays dark in both themes for a consistent brand look */
section[data-testid="stSidebar"] {{
    background-color: {PRIMARY_DARK};
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT_LIGHT} !important;
}}
section[data-testid="stSidebar"] .stRadio > div {{
    gap: 2px;
}}
section[data-testid="stSidebar"] label {{
    padding: 6px 8px;
    border-radius: 6px;
}}

/* KPI cards */
.kpi-card {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}}
.kpi-label {{
    font-size: 13px;
    color: {TEXT_MUTED};
    font-weight: 500;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 700;
    color: {TEXT_MAIN};
}}
.kpi-sub {{
    font-size: 12px;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}

/* Section headers */
.section-title {{
    font-size: 20px;
    font-weight: 700;
    color: {TEXT_MAIN};
    margin-top: 8px;
    margin-bottom: 4px;
}}
.section-sub {{
    font-size: 14px;
    color: {TEXT_SUBTLE};
    margin-bottom: 18px;
}}

/* Risk badges (self-contained pastel bg + dark text, same in both themes) */
.risk-badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.02em;
}}
.risk-low {{ background-color: #EAF3DE; color: #27500A; }}
.risk-medium {{ background-color: #FAEEDA; color: #854F0B; }}
.risk-high {{ background-color: #FCEBEB; color: #791F1F; }}

/* Chart / content cards - built with st.container(key="card_...") so
   content is properly nested inside the styled box */
div[class*="st-key-card_"] {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 20px 22px;
}}

hr {{ border-color: {BORDER_COLOR}; }}

/* Force readable text everywhere in the main content area, matching the
   selected theme regardless of the visitor's OS/browser dark-mode setting */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
    color: {TEXT_MAIN};
}}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    color: {TEXT_MAIN} !important;
}}
.stDataFrame, .stDataFrame * {{
    color: {TEXT_MAIN} !important;
}}
.stTable, .stTable table, .stTable th, .stTable td {{
    color: {TEXT_MAIN} !important;
    background-color: {CARD_BG} !important;
    border-color: {BORDER_COLOR} !important;
}}
.stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label,
.stTextInput label, .stRadio label {{
    color: {TEXT_MAIN} !important;
}}
[data-testid="stExpander"] {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_COLORS = {"Yes": ACCENT_RED, "No": ACCENT_TEAL}

# Apply stored filters to the dataframe used by every page
filters = st.session_state.get("filters", {})
df = df_full.copy()
for col, selected in filters.items():
    if selected:
        df = df[df[col].isin(selected)]

if df.empty:
    st.warning("No employees match the selected filters. Showing full dataset instead.")
    df = df_full.copy()


def render_chart(fig):
    """Renders a Plotly figure with a transparent background so it always
    blends seamlessly with the surrounding card, regardless of the active
    light/dark theme, and forces readable font/axis colors."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_MAIN,
        legend_font_color=TEXT_MAIN,
    )
    fig.update_xaxes(color=TEXT_MAIN, gridcolor=BORDER_COLOR)
    fig.update_yaxes(color=TEXT_MAIN, gridcolor=BORDER_COLOR)
    st.plotly_chart(fig, width='stretch')


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if page == "Overview":
    st.markdown('<div class="section-title">HR Employee Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Analyze employee data, understand attrition trends and '
        'predict future risks.</div>',
        unsafe_allow_html=True,
    )

    kpis = calculate_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Employees", f"{kpis['total_employees']:,}")
    with c2:
        kpi_card("Attrition Rate", f"{kpis['attrition_rate']}%", f"{kpis['employees_left']} employees left")
    with c3:
        kpi_card("Average Salary", f"₹{kpis['avg_monthly_income']:,.0f}")
    with c4:
        kpi_card("Average Experience", f"{kpis['avg_experience']} yrs")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        with st.container(key="card_1", border=False):
            st.markdown("**Department Workforce**")
            dept_counts = df["Department"].value_counts().reset_index()
            dept_counts.columns = ["Department", "Employees"]
            fig = px.pie(dept_counts, names="Department", values="Employees", hole=0.55,
                         color_discrete_sequence=["#1D9E75", "#378ADD", "#D85A30"])
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340,
                               showlegend=False)
            render_chart(fig)

    with col2:
        with st.container(key="card_2", border=False):
            st.markdown("**Attrition by Department**")
            dept_attr = attrition_rate_by(df, "Department")
            fig = px.bar(dept_attr, x="Department", y="AttritionRate",
                         text="AttritionRate", color_discrete_sequence=[ACCENT_RED])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340,
                               yaxis_title="Attrition Rate (%)", xaxis_title="")
            render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(key="card_3", border=False):
        st.markdown("**Key HR Metrics**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average Age", f"{kpis['avg_age']} yrs")
        m2.metric("Avg Job Satisfaction", f"{kpis['avg_job_satisfaction']} / 4")
        m3.metric("Avg Training / Year", f"{kpis['avg_training_hours']}")
        m4.metric("Employees Left", f"{kpis['employees_left']}")


# ===========================================================================
# PAGE: EMPLOYEE ANALYSIS
# ===========================================================================
elif page == "Employee Analysis":
    st.markdown('<div class="section-title">Employee Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Workforce composition and demographics.</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(key="card_4", border=False):
            st.markdown("**Age Group Distribution**")
            age_counts = df["AgeGroup"].value_counts().sort_index().reset_index()
            age_counts.columns = ["AgeGroup", "Employees"]
            fig = px.bar(age_counts, x="AgeGroup", y="Employees",
                         color_discrete_sequence=["#378ADD"])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, xaxis_title="")
            render_chart(fig)

    with col2:
        with st.container(key="card_5", border=False):
            st.markdown("**Gender Split**")
            gender_counts = df["Gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Employees"]
            fig = px.pie(gender_counts, names="Gender", values="Employees", hole=0.55,
                         color_discrete_sequence=["#7F77DD", "#D4537E"])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        with st.container(key="card_6", border=False):
            st.markdown("**Salary Range Distribution**")
            sal_counts = df["SalaryRange"].value_counts().sort_index().reset_index()
            sal_counts.columns = ["SalaryRange", "Employees"]
            fig = px.bar(sal_counts, x="SalaryRange", y="Employees",
                         color_discrete_sequence=["#1D9E75"])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, xaxis_title="")
            render_chart(fig)

    with col4:
        with st.container(key="card_7", border=False):
            st.markdown("**Job Role Distribution**")
            role_counts = df["JobRole"].value_counts().reset_index()
            role_counts.columns = ["JobRole", "Employees"]
            fig = px.bar(role_counts, x="Employees", y="JobRole", orientation="h",
                         color_discrete_sequence=["#BA7517"])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, yaxis_title="")
            render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(key="card_8", border=False):
        st.markdown("**Years at Company Distribution**")
        yrs_counts = df["YearsAtCompanyGroup"].value_counts().sort_index().reset_index()
        yrs_counts.columns = ["YearsAtCompany", "Employees"]
        fig = px.bar(yrs_counts, x="YearsAtCompany", y="Employees",
                     color_discrete_sequence=["#378ADD"])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, xaxis_title="")
        render_chart(fig)


# ===========================================================================
# PAGE: ATTRITION ANALYSIS
# ===========================================================================
elif page == "Attrition Analysis":
    st.markdown('<div class="section-title">Attrition Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What factors are associated with employees leaving?</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(key="card_9", border=False):
            st.markdown("**Attrition by Salary Range**")
            sal_attr = attrition_rate_by(df, "SalaryRange").sort_values("SalaryRange")
            fig = px.pie(sal_attr, names="SalaryRange", values="Left", hole=0.5,
                         color_discrete_sequence=px.colors.sequential.Reds_r)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            render_chart(fig)

    with col2:
        with st.container(key="card_10", border=False):
            st.markdown("**Attrition by Years at Company**")
            yrs_attr = attrition_rate_by(df, "YearsAtCompanyGroup").sort_values("YearsAtCompanyGroup")
            fig = px.bar(yrs_attr, x="YearsAtCompanyGroup", y="AttritionRate",
                         text="AttritionRate", color_discrete_sequence=[ACCENT_RED])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               yaxis_title="Attrition Rate (%)", xaxis_title="")
            render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        with st.container(key="card_11", border=False):
            st.markdown("**Job Satisfaction vs Attrition**")
            sat = df.groupby(["JobSatisfaction", "Attrition"], observed=True).size().reset_index(name="Count")
            fig = px.bar(sat, x="JobSatisfaction", y="Count", color="Attrition", barmode="group",
                         color_discrete_map=CHART_COLORS)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               xaxis_title="Job Satisfaction (1-4)")
            render_chart(fig)

    with col4:
        with st.container(key="card_12", border=False):
            st.markdown("**Overtime vs Attrition**")
            ot_attr = attrition_rate_by(df, "OverTime")
            fig = px.bar(ot_attr, x="OverTime", y="AttritionRate", text="AttritionRate",
                         color_discrete_sequence=[ACCENT_AMBER])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               yaxis_title="Attrition Rate (%)")
            render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        with st.container(key="card_13", border=False):
            st.markdown("**Age Group vs Attrition Rate**")
            age_attr = attrition_rate_by(df, "AgeGroup").sort_values("AgeGroup")
            fig = px.bar(age_attr, x="AgeGroup", y="AttritionRate", text="AttritionRate",
                         color_discrete_sequence=["#D4537E"])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               yaxis_title="Attrition Rate (%)")
            render_chart(fig)

    with col6:
        with st.container(key="card_14", border=False):
            st.markdown("**Job Role vs Attrition Rate**")
            role_attr = attrition_rate_by(df, "JobRole")
            fig = px.bar(role_attr, x="AttritionRate", y="JobRole", orientation="h",
                         text="AttritionRate", color_discrete_sequence=[ACCENT_RED])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               xaxis_title="Attrition Rate (%)", yaxis_title="")
            render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(key="card_15", border=False):
        st.markdown("**Attrition Heatmap: Job Role x Department (%)**")
        heatmap_data = attrition_heatmap_data(df)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Reds",
            text=heatmap_data.values,
            texttemplate="%{text}%",
            hovertemplate="Job Role: %{y}<br>Department: %{x}<br>Attrition: %{z}%<extra></extra>",
        ))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420)
        render_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        with st.container(key="card_16", border=False):
            st.markdown("**Attrition by Promotion History**")
            promo_attr = attrition_rate_by(df, "PromotionLast5Years")
            promo_attr["PromotionLast5Years"] = promo_attr["PromotionLast5Years"].map(
                {1: "Promoted (last 5 yrs)", 0: "Not promoted"})
            fig = px.bar(promo_attr, x="PromotionLast5Years", y="AttritionRate", text="AttritionRate",
                         color_discrete_sequence=["#853F0B" if False else ACCENT_AMBER])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               yaxis_title="Attrition Rate (%)", xaxis_title="")
            render_chart(fig)

    with col8:
        with st.container(key="card_17", border=False):
            st.markdown("**Attrition by Business Travel**")
            travel_attr = attrition_rate_by(df, "BusinessTravel")
            fig = px.bar(travel_attr, x="BusinessTravel", y="AttritionRate", text="AttritionRate",
                         color_discrete_sequence=["#378ADD"])
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                               yaxis_title="Attrition Rate (%)")
            render_chart(fig)


# ===========================================================================
# PAGE: PREDICTION
# ===========================================================================
elif page == "Prediction":
    st.markdown('<div class="section-title">Attrition Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Enter an employee profile to estimate attrition risk using '
        'the trained machine-learning model.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This is a predictive-analytics demonstration built on synthetic data. "
        "It is **not** a definitive HR decision-making system and should never be used "
        "as the sole basis for real employment decisions.",
        icon=None,
    )

    if not model_available():
        st.error("No trained model found. Run `python src/prediction.py` first.")
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                employee_id = st.text_input("Employee ID", value="EMP00000")
                age = st.number_input("Age", 18, 65, 30)
                department = st.selectbox("Department", sorted(df_full["Department"].unique()))
                job_role_opts = sorted(
                    df_full[df_full["Department"] == department]["JobRole"].unique()
                ) if department in df_full["Department"].values else sorted(df_full["JobRole"].unique())
                job_role = st.selectbox("Job Role", job_role_opts)
                monthly_income = st.number_input("Monthly Income (₹)", 10000, 300000, 45000, step=1000)

            with c2:
                years_at_company = st.number_input("Years at Company", 0, 40, 3)
                years_in_role = st.number_input("Years in Current Role", 0, 40, 2)
                job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
                overtime = st.selectbox("OverTime", ["No", "Yes"])

            with c3:
                job_level = st.slider("Job Level (1-5)", 1, 5, 2)
                business_travel = st.selectbox(
                    "Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1
                )
                distance_from_home = st.number_input("Distance From Home (km)", 1, 60, 10)
                years_since_promotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
                performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)

            predict_clicked = st.form_submit_button("Predict Attrition Risk", width='stretch')

        if predict_clicked:
            employee = {
                "Age": age,
                "Education": 3,
                "JobLevel": job_level,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_role,
                "YearsSinceLastPromotion": years_since_promotion,
                "TotalWorkingYears": max(years_at_company, age - 22),
                "MonthlyIncome": monthly_income,
                "HourlyRate": 60,
                "JobSatisfaction": job_satisfaction,
                "EnvironmentSatisfaction": 3,
                "RelationshipSatisfaction": 3,
                "WorkLifeBalance": work_life_balance,
                "PerformanceRating": performance_rating,
                "TrainingTimesLastYear": 2,
                "NumCompaniesWorked": 2,
                "DistanceFromHome": distance_from_home,
                "JobInvolvement": 3,
                "StockOptionLevel": 1,
                "PromotionLast5Years": 1 if years_since_promotion <= 5 else 0,
                "Department": department,
                "JobRole": job_role,
                "EducationField": "Life Sciences",
                "OverTime": overtime,
                "BusinessTravel": business_travel,
                "EmploymentType": "Full-Time",
            }

            result = predict_single_employee(employee)
            risk_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[result["risk_level"]]

            st.markdown("<br>", unsafe_allow_html=True)
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                with st.container(key="card_18", border=False):
                    st.markdown(f"**Employee:** {employee_id}")
                    st.markdown(f"<div class='kpi-value'>{result['attrition_probability']}%</div>",
                                unsafe_allow_html=True)
                    st.markdown(f"<span class='risk-badge {risk_class}'>{result['risk_level']} RISK</span>",
                                unsafe_allow_html=True)

            with rc2:
                with st.container(key="card_19", border=False):
                    st.markdown("**Most Influential Factors (model-wide)**")
                    if result["top_factors"]:
                        factors_df = pd.DataFrame(result["top_factors"], columns=["Feature", "Importance"])
                        fig = px.bar(factors_df.sort_values("Importance"), x="Importance", y="Feature",
                                     orientation="h", color_discrete_sequence=[ACCENT_TEAL])
                        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240)
                        render_chart(fig)
                    else:
                        st.caption("Feature importance is not available for this model type.")


# ===========================================================================
# PAGE: SQL INSIGHTS
# ===========================================================================
elif page == "SQL Insights":
    st.markdown('<div class="section-title">SQL Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">HR metrics queried directly from the SQLite database.</div>',
                unsafe_allow_html=True)

    if not os.path.exists(DB_PATH):
        st.error("Database not found. Run `python src/database.py` first.")
    else:
        queries = {
            "Department employee count": """
                SELECT Department, COUNT(*) AS employee_count
                FROM employees
                GROUP BY Department
                ORDER BY employee_count DESC;
            """,
            "Department attrition rate": """
                SELECT Department,
                       COUNT(*) AS total_employees,
                       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS employees_left,
                       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
                FROM employees
                GROUP BY Department
                ORDER BY attrition_rate_pct DESC;
            """,
            "Average salary by department": """
                SELECT Department, ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
                FROM employees
                GROUP BY Department
                ORDER BY avg_monthly_income DESC;
            """,
            "Average experience by department": """
                SELECT Department, ROUND(AVG(TotalWorkingYears), 1) AS avg_total_working_years
                FROM employees
                GROUP BY Department
                ORDER BY avg_total_working_years DESC;
            """,
            "High-risk segments (overtime + low satisfaction + low WLB)": """
                SELECT Department, JobRole, COUNT(*) AS at_risk_employee_count,
                       ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
                FROM employees
                WHERE OverTime = 'Yes' AND JobSatisfaction <= 2 AND WorkLifeBalance <= 2
                GROUP BY Department, JobRole
                ORDER BY at_risk_employee_count DESC;
            """,
        }

        for i, (title, query) in enumerate(queries.items()):
            with st.container(key=f"card_20_{i}", border=False):
                st.markdown(f"**{title}**")
                result_df = run_query(query)
                # st.table renders a plain HTML table so it always follows the
                # selected light/dark theme (the interactive st.dataframe grid
                # is canvas-rendered and would ignore our custom CSS toggle).
                st.table(result_df.style.hide(axis="index"))
                with st.expander("View SQL query"):
                    st.code(query.strip(), language="sql")
            st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# PAGE: ABOUT
# ===========================================================================
elif page == "About":
    st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)

    with st.container(key="card_21", border=False):
        st.markdown(
"""**Project Name:** HR Employee Analytics & Attrition Prediction Dashboard

**Purpose:** Analyze employee data, surface factors associated with attrition,
and demonstrate an end-to-end analytics + machine-learning workflow, from raw
data through cleaning, EDA, SQL analysis, KPI reporting, and predictive modeling.

**Technology Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn,
SQLite, Streamlit, Plotly, Joblib.

**Dataset Information:** 1,200 synthetic employee records generated with
realistic (but not perfectly deterministic) relationships between attrition and
factors such as overtime, job satisfaction, work-life balance, income, tenure,
commute distance, promotion history, and business travel. The dataset is
**not** real company data.

**Machine Learning Approach:** Logistic Regression and Random Forest classifiers
were trained and compared on standard classification metrics (accuracy,
precision, recall, F1, ROC-AUC). The model with the higher ROC-AUC on a held-out
test set was selected. Gender and marital status were intentionally excluded
from the model's input features to avoid using protected attributes in
attrition predictions.

**Limitations:**
- The dataset is synthetic; real-world HR data is noisier and more complex.
- The model's precision/recall reflect realistic, imperfect performance -
  it is a demonstration of the workflow, not a production-grade system.
- Predictions should never be used as the sole basis for real employment
  decisions. They are one input among many for HR professionals to consider.
- The model was not audited for fairness across all possible subgroups.
"""
        )
        st.markdown("Built with Python, Pandas, SQL, Scikit-learn and Streamlit.")
