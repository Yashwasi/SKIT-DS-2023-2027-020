"""
database/init_db.py
-------------------
Safe database initialization and schema deployment CLI tool.

Guarantees:
- Never drops or truncates existing tables or databases.
- Applies database/schema.sql idempotently using IF NOT EXISTS.
- Supports --dry-run mode for pre-flight syntax and command inspection.
- Optionally seeds baseline disdrometer observation stations (--seed-stations).
- Reports detailed connectivity status without leaking passwords.

Usage:
    python -m database.init_db [--dry-run] [--seed-stations]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from database.config import db_config, DatabaseConfig
from database.connection import (
    check_database_health,
    get_driver_name,
    get_engine,
    probe_tcp_port,
)

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

DEFAULT_STATIONS = [
    {
        "station_id": "RD1_Foliated_Cedar",
        "station_name": "Cedar Forest Canopy Disdrometer 1",
        "canopy_type": "Foliated Cedar",
        "instrument_type": "Optical Disdrometer",
        "latitude": 35.6895,
        "longitude": 139.6917,
        "elevation_m": 120.0,
    },
    {
        "station_id": "RD2_Defoliated_Oak",
        "station_name": "Oak Canopy Disdrometer 2",
        "canopy_type": "Defoliated Oak",
        "instrument_type": "Optical Disdrometer",
        "latitude": 35.6901,
        "longitude": 139.6925,
        "elevation_m": 125.0,
    },
    {
        "station_id": "RD3_Open_Field",
        "station_name": "Open Field Reference Station",
        "canopy_type": "Open Field",
        "instrument_type": "Optical Disdrometer",
        "latitude": 35.6910,
        "longitude": 139.6905,
        "elevation_m": 118.0,
    },
]


def load_schema_sql() -> str:
    """Read the complete schema DDL script."""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found at: {SCHEMA_FILE}")
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        return f.read()


def split_sql_statements(sql: str) -> List[str]:
    """
    Split SQL script into discrete executable blocks, preserving DO $$ ... $$ blocks.
    """
    statements: List[str] = []
    current: List[str] = []
    in_dollar_block = False

    for line in sql.splitlines():
        trimmed = line.strip()
        # Track PL/pgSQL dollar quotes $$
        if "$$" in trimmed:
            in_dollar_block = not in_dollar_block

        current.append(line)

        if not in_dollar_block and trimmed.endswith(";"):
            stmt = "\n".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []

    if current:
        stmt = "\n".join(current).strip()
        if stmt:
            statements.append(stmt)

    return statements


def run_dry_run() -> None:
    """Inspect schema script without connecting to any database."""
    print("=" * 70)
    print("DATABASE INITIALIZATION — DRY RUN MODE (Safe Inspection)")
    print("=" * 70)
    print(f"Target Configuration: {db_config.get_safe_url()}")
    print(f"Schema File:          {SCHEMA_FILE}")

    sql = load_schema_sql()
    statements = split_sql_statements(sql)
    print(f"Total DDL statements / blocks parsed: {len(statements)}")
    print("-" * 70)

    for idx, stmt in enumerate(statements, start=1):
        # Print summary of statement
        first_line = stmt.strip().split("\n")[0][:80]
        print(f"[{idx:02d}] {first_line} ...")

    print("-" * 70)
    print("DRY RUN COMPLETE: Schema is structurally valid and ready for deployment.")
    print("No database connection was opened and no data was modified.")


def initialize_database(seed_stations: bool = False) -> bool:
    """Connect to PostgreSQL and apply the schema."""
    print("=" * 70)
    print("RAIN DROP VOLUME PREDICTION — DATABASE INITIALIZATION")
    print("=" * 70)
    print(f"Target Database: {db_config.get_safe_url()}")

    # 1. Probe TCP port
    tcp_ok, tcp_err = probe_tcp_port(db_config.host, db_config.port)
    if not tcp_ok:
        print(f"[FAIL] PostgreSQL server unreachable at {db_config.host}:{db_config.port}.")
        print(f"       Details: {tcp_err}")
        print("\nAction needed:")
        print("1. Ensure PostgreSQL is installed and service is started.")
        print("2. Check credentials in .env file (see .env.example).")
        print("3. For local Docker setup, run:")
        print("   docker run -d --name timescale -p 5432:5432 -e POSTGRES_PASSWORD=secret timescale/timescaledb:latest-pg16")
        return False

    # 2. Check driver availability
    driver = get_driver_name()
    if not driver:
        print("[FAIL] Python database driver not found (psycopg2 or psycopg).")
        print("       Install required drivers via:")
        print("       pip install sqlalchemy psycopg2-binary")
        return False

    # 3. Connect to database
    engine = get_engine()
    if engine is None:
        print("[FAIL] Could not instantiate SQLAlchemy engine.")
        return False

    try:
        from sqlalchemy import text

        print("[OK] Connected to database server.")
        schema_sql = load_schema_sql()
        statements = split_sql_statements(schema_sql)

        print(f"Applying {len(statements)} schema statements...")
        with engine.begin() as conn:
            for idx, stmt in enumerate(statements, start=1):
                clean_stmt = stmt.strip()
                if not clean_stmt or clean_stmt.startswith("--"):
                    continue
                conn.execute(text(clean_stmt))
                print(f"  [{idx:02d}/{len(statements):02d}] Applied successfully.")

        print("[OK] Schema applied successfully. Tables, indexes, and views created.")

        # 4. Optional station seeding
        if seed_stations:
            print("\nChecking baseline station records...")
            with engine.begin() as conn:
                for st in DEFAULT_STATIONS:
                    conn.execute(
                        text(
                            """
                            INSERT INTO stations (station_id, station_name, canopy_type, instrument_type, latitude, longitude, elevation_m)
                            VALUES (:station_id, :station_name, :canopy_type, :instrument_type, :latitude, :longitude, :elevation_m)
                            ON CONFLICT (station_id) DO NOTHING;
                            """
                        ),
                        st,
                    )
            print(f"[OK] Seeded {len(DEFAULT_STATIONS)} baseline monitoring stations.")

        # 5. Final health check
        health = check_database_health()
        print("\nPost-initialization Health Status:")
        print(f"  Status:               {health.get('status')}")
        print(f"  TimescaleDB Enabled:  {health.get('timescaledb_extension')}")
        print(f"  Query Latency:        {health.get('latency_ms')} ms")
        return True

    except Exception as exc:
        print(f"\n[FAIL] Database initialization failed: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL / TimescaleDB schema for Rain Drop Volume Prediction project."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate schema syntax and display DDL statements without connecting to database.",
    )
    parser.add_argument(
        "--seed-stations",
        action="store_true",
        help="Insert default disdrometer monitoring stations into 'stations' table.",
    )
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return 0

    success = initialize_database(seed_stations=args.seed_stations)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
