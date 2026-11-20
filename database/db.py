import os, sqlite3
import pandas as pd

HERE = os.path.dirname(__file__)
DB = os.path.join(HERE, "artistiq.db")
CSV = os.path.join(HERE, "..", "dataset", "mxmh_survey_results.csv")

def save(df, table="survey"):
    with sqlite3.connect(DB) as c:
        df.to_sql(table, c, if_exists="replace", index=False)

def load(table="survey"):
    """Load from SQLite; on first run seed it from the bundled CSV."""
    try:
        with sqlite3.connect(DB) as c:
            return pd.read_sql(f"select * from {table}", c)
    except Exception:
        df = pd.read_csv(CSV); save(df, table); return df

def reset():
    if os.path.exists(DB): os.remove(DB)
