-- =====================================================================
-- hr_queries.sql
-- HR Employee Analytics - SQL Analysis
--
-- These queries run against the "employees" table created by
-- src/database.py (data/hr_database.db). They can be run with any
-- SQLite client, e.g.:
--
--     sqlite3 data/hr_database.db < sql/hr_queries.sql
--
-- Several of these queries are also executed live from the Streamlit
-- "SQL Insights" page (dashboard/app.py) via src/database.run_query().
-- =====================================================================


-- 1. Total number of employees in the company
SELECT COUNT(*) AS total_employees
FROM employees;


-- 2. Total employees by department
SELECT
    Department,
    COUNT(*) AS employee_count
FROM employees
GROUP BY Department
ORDER BY employee_count DESC;


-- 3. Total number of employees who left (Attrition = 'Yes')
SELECT COUNT(*) AS employees_left
FROM employees
WHERE Attrition = 'Yes';


-- 4. Overall attrition rate (%)
SELECT
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) AS attrition_rate_pct
FROM employees;


-- 5. Average salary by department
SELECT
    Department,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
FROM employees
GROUP BY Department
ORDER BY avg_monthly_income DESC;


-- 6. Average experience (total working years) by department
SELECT
    Department,
    ROUND(AVG(TotalWorkingYears), 1) AS avg_total_working_years
FROM employees
GROUP BY Department
ORDER BY avg_total_working_years DESC;


-- 7. Departments with the highest attrition rate
SELECT
    Department,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS employees_left,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;


-- 8. Job roles with the highest attrition rate
SELECT
    JobRole,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS employees_left,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY JobRole
ORDER BY attrition_rate_pct DESC;


-- 9. Employees currently working overtime
SELECT COUNT(*) AS overtime_employee_count
FROM employees
WHERE OverTime = 'Yes';


-- 10. Employees with low job satisfaction (score of 1 or 2 out of 4)
SELECT COUNT(*) AS low_job_satisfaction_count
FROM employees
WHERE JobSatisfaction <= 2;


-- 11. Employees who have NOT received a promotion in the last 5 years
SELECT COUNT(*) AS no_recent_promotion_count
FROM employees
WHERE PromotionLast5Years = 0;


-- 12. Salary comparison: employees who stayed vs. employees who left
SELECT
    Attrition,
    COUNT(*) AS employee_count,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(MIN(MonthlyIncome), 0) AS min_monthly_income,
    ROUND(MAX(MonthlyIncome), 0) AS max_monthly_income
FROM employees
GROUP BY Attrition;


-- 13. Attrition by age group
SELECT
    CASE
        WHEN Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN Age BETWEEN 36 AND 45 THEN '36-45'
        WHEN Age BETWEEN 46 AND 55 THEN '46-55'
        ELSE '56+'
    END AS age_group,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS employees_left,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY age_group
ORDER BY age_group;


-- 14. Attrition by experience group (years at company)
SELECT
    CASE
        WHEN YearsAtCompany BETWEEN 0 AND 1 THEN '0-1 yrs'
        WHEN YearsAtCompany BETWEEN 2 AND 3 THEN '2-3 yrs'
        WHEN YearsAtCompany BETWEEN 4 AND 6 THEN '4-6 yrs'
        WHEN YearsAtCompany BETWEEN 7 AND 10 THEN '7-10 yrs'
        ELSE '10+ yrs'
    END AS experience_group,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS employees_left,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY experience_group
ORDER BY attrition_rate_pct DESC;


-- 15. Top "high risk" segments: overtime + low satisfaction + low work-life balance
--     (a simple rule-based risk segment, separate from the ML model)
SELECT
    Department,
    JobRole,
    COUNT(*) AS at_risk_employee_count,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
FROM employees
WHERE OverTime = 'Yes'
  AND JobSatisfaction <= 2
  AND WorkLifeBalance <= 2
GROUP BY Department, JobRole
ORDER BY at_risk_employee_count DESC;
