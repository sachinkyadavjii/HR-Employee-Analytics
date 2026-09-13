"""
database.py
------------
Loads the cleaned HR dataset into a local SQLite database so it can be
queried with plain SQL (see sql/hr_queries.sql) and reused by the
Streamlit dashboard's "SQL Insights" page.

Run directly to (re)build the database:
    python src/database.py
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "hr_employee_data_clean.csv")
RAW_CSV = os.path.join(BASE_DIR, "data", "hr_employee_data.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "hr_database.db")
TABLE_NAME = "employees"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection, creating the file if needed."""
    return sqlite3.connect(db_path)


def load_csv_to_db(csv_path: str = None, db_path: str = DB_PATH,
                    table_name: str = TABLE_NAME) -> None:
    """Loads a CSV file into a SQLite table (overwrites if it exists)."""
    if csv_path is None:
        csv_path = CLEAN_CSV if os.path.exists(CLEAN_CSV) else RAW_CSV

    df = pd.read_csv(csv_path)
    conn = get_connection(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        print(f"Loaded {len(df):,} rows into '{table_name}' table -> {db_path}")
    finally:
        conn.close()


def run_query(query: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Runs a SQL query against the HR database and returns a DataFrame."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    load_csv_to_db()

    # Quick smoke test
    test_df = run_query(f"SELECT COUNT(*) AS total_employees FROM {TABLE_NAME}")
    print(test_df)
