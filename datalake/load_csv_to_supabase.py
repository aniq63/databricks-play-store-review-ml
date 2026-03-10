"""
Load reviews.csv into Supabase PostgreSQL as the 'raw_reviews' table.

Usage:
    python database/load_csv_to_supabase.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ── 1. Load environment ───────────────────────────────────────────────────────
load_dotenv()

raw_url = os.getenv("DATABASE_URL", "")
if not raw_url:
    raise ValueError("DATABASE_URL not found in .env")

# SQLAlchemy sync engine needs postgresql+psycopg2://
if raw_url.startswith("postgres://"):
    db_url = raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif raw_url.startswith("postgresql://"):
    db_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    db_url = raw_url

# Supabase transaction pooler requires this
engine = create_engine(
    db_url,
    connect_args={"options": "-c statement_timeout=60000"},
    pool_pre_ping=True,
)

# ── 2. Load CSV ───────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "raw_data", "reviews.csv")
CSV_PATH = os.path.normpath(CSV_PATH)

print(f"📂 Reading CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"✅ Loaded {len(df):,} rows  |  Columns: {list(df.columns)}")
print(df.head(3))

# ── 3. Write to Supabase ──────────────────────────────────────────────────────
TABLE_NAME = "raw_reviews"

print(f"\nWriting to Supabase table '{TABLE_NAME}' ...")

with engine.begin() as conn:
    # Drop & recreate for a clean load
    conn.execute(text(f'DROP TABLE IF EXISTS "{TABLE_NAME}"'))

df.to_sql(
    name=TABLE_NAME,
    con=engine,
    if_exists="replace",   # creates table automatically from df schema
    index=False,
    chunksize=500,         # batch inserts for speed
    method="multi",
)

print(f"Done! Table '{TABLE_NAME}' created in Supabase with {len(df):,} rows.")

# ── 4. Quick verify ───────────────────────────────────────────────────────────
with engine.connect() as conn:
    result = conn.execute(text(f'SELECT COUNT(*) FROM "{TABLE_NAME}"'))
    count = result.scalar()
    print(f"Row count in Supabase: {count:,}")
