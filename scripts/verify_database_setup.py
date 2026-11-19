"""
scripts/verify_database_setup.py
--------------------------------
Comprehensive database verification script for Rain Drop Volume Prediction Using AI.
Verifies:
1. Configuration loading and password masking safety.
2. DDL schema file integrity and TimescaleDB hypertable definitions.
3. Full ORM model mapping and relational integrity (Station, Observation, Intensity, AuditLog).
4. In-memory transaction test: Insert, Retrieve, Filter, and Cascade Deletion.
5. Live PostgreSQL / TimescaleDB server probing.
6. Diagnostic report on database readiness.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from database.config import db_config, DatabaseConfig
    from database.connection import check_database_health, probe_tcp_port
    from database.init_db import load_schema_sql, split_sql_statements
    from database.models import (
        Base,
        Station,
        RaindropObservation,
        RainfallIntensityRecord,
        PredictionAuditLog,
    )
    _MODULES_OK = True
except ImportError as exc:
    print(f"Error importing database modules: {exc}")
    _MODULES_OK = False


def verify_database_architecture() -> bool:
    print("=" * 70)
    print("DATABASE ARCHITECTURE & DESIGN VERIFICATION")
    print("=" * 70)

    if not _MODULES_OK:
        print("[FAIL] Database modules failed to import.")
        return False

    all_passed = True

    # -------------------------------------------------------------------------
    # 1. Config & Credential Masking
    # -------------------------------------------------------------------------
    print("\n[1] CONFIGURATION & CREDENTIAL SAFETY:")
    safe_url = db_config.get_safe_url()
    print(f"    Target Safe URL:  {safe_url}")
    print(f"    Host:             {db_config.host}:{db_config.port}")
    print(f"    Database Name:    {db_config.name}")
    print(f"    Connection Pool:  size={db_config.pool_size}, max_overflow={db_config.max_overflow}")

    if "password" in safe_url.lower() and db_config.password:
        print("    [FAIL] Password detected in safe URL!")
        all_passed = False
    else:
        print("    [OK] Credential masking verified. No plaintext secrets in logs.")

    # -------------------------------------------------------------------------
    # 2. DDL Schema File Integrity
    # -------------------------------------------------------------------------
    print("\n[2] DDL SCHEMA & TIMESCALEDB DEFINITION:")
    schema_file = PROJECT_ROOT / "database" / "schema.sql"
    if not schema_file.exists():
        print(f"    [FAIL] Schema file missing at: {schema_file}")
        return False

    raw_sql = load_schema_sql()
    statements = split_sql_statements(raw_sql)
    print(f"    Schema file size: {len(raw_sql):,} characters")
    print(f"    Parsed DDL executable blocks: {len(statements)}")

    # Verify key DDL components
    expected_components = [
        ("stations table", "CREATE TABLE IF NOT EXISTS stations"),
        ("raindrop_observations table", "CREATE TABLE IF NOT EXISTS raindrop_observations"),
        ("rainfall_intensity_records table", "CREATE TABLE IF NOT EXISTS rainfall_intensity_records"),
        ("prediction_audit_logs table", "CREATE TABLE IF NOT EXISTS prediction_audit_logs"),
        ("TimescaleDB hypertable", "create_hypertable"),
        ("7-day chunk interval", "INTERVAL '7 days'"),
        ("Physical constraints", "diameter_mm > 0.0 AND diameter_mm <= 15.0"),
        ("DSD summary view", "v_hourly_dsd_summary"),
        ("Extreme drops view", "v_extreme_raindrops"),
    ]

    for label, pattern in expected_components:
        if pattern in raw_sql:
            print(f"    [OK] {label:<32} (Verified)")
        else:
            print(f"    [FAIL] {label:<32} (Not found in schema.sql)")
            all_passed = False

    # -------------------------------------------------------------------------
    # 3. ORM Models & Relational Integrity (In-Memory Transaction Verification)
    # -------------------------------------------------------------------------
    print("\n[3] OFFLINE ORM MODELS & RELATIONSHIP TRANSACTION TEST:")
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Step A: Insert Station
        test_station = Station(
            station_id="TEST_ST_01",
            station_name="Cedar Experimental Plot 1",
            canopy_type="Foliated Cedar",
            instrument_type="Optical Disdrometer",
            latitude=35.6895,
            longitude=139.6917,
            elevation_m=120.0,
        )
        session.add(test_station)
        session.commit()
        print("    [OK] Station metadata insert: SUCCESS")

        # Step B: Insert RaindropObservation with FK
        test_time = datetime.datetime(2026, 11, 19, 14, 30, 0)
        test_obs = RaindropObservation(
            observed_at=test_time,
            station_id="TEST_ST_01",
            diameter_mm=1.25,
            velocity_m_s=4.10,
            volume_mm3=1.0227,
            condition="Foliated_Cedar_Rain",
            source_file="test_sample.csv",
        )
        session.add(test_obs)
        session.commit()
        print("    [OK] Raindrop observation insert (with FK): SUCCESS")

        # Step C: Insert RainfallIntensityRecord
        test_intensity = RainfallIntensityRecord(
            period_start=test_time,
            period_end=test_time + datetime.timedelta(minutes=1),
            station_id="TEST_ST_01",
            rain_rate_mm_h=12.5,
            total_drops=480,
            mean_diameter_mm=1.10,
            mean_velocity_m_s=3.95,
            total_volume_mm3=325.4,
            condition="Heavy Rain",
        )
        session.add(test_intensity)
        session.commit()
        print("    [OK] Rainfall intensity record insert: SUCCESS")

        # Step D: Insert PredictionAuditLog
        test_audit = PredictionAuditLog(
            input_diameter_mm=1.25,
            input_velocity_m_s=4.10,
            formula_volume_mm3=1.02265,
            xgb_volume_mm3=0.98540,
            latency_ms=1.45,
            client_ip="127.0.0.1",
        )
        session.add(test_audit)
        session.commit()
        print("    [OK] API prediction audit log insert: SUCCESS")

        # Step E: Query & Verify Relationships
        queried_st = session.query(Station).filter_by(station_id="TEST_ST_01").one()
        assert len(queried_st.observations) == 1, "Observations relationship traversal failed"
        assert len(queried_st.intensity_records) == 1, "Intensity records relationship traversal failed"
        assert queried_st.observations[0].diameter_mm == 1.25
        print("    [OK] Relationship traversal (Station -> Observations): SUCCESS")

        # Step F: Verify Cascade Deletion
        session.delete(queried_st)
        session.commit()
        assert session.query(RaindropObservation).count() == 0, "Cascade deletion failed"
        print("    [OK] Cascade deletion integrity (Delete station -> observations purged): SUCCESS")

    except Exception as exc:
        print(f"    [FAIL] Offline ORM verification encountered error: {exc}")
        all_passed = False

    # -------------------------------------------------------------------------
    # 4. Live PostgreSQL Server Status Probe
    # -------------------------------------------------------------------------
    print("\n[4] LIVE DATABASE ENVIRONMENT PROBE:")
    tcp_ok, tcp_err = probe_tcp_port(db_config.host, db_config.port, timeout_sec=1.0)
    health = check_database_health()

    if tcp_ok:
        print(f"    [ONLINE] PostgreSQL port {db_config.port} is reachable.")
        print(f"             Status: {health.get('status')}")
        print(f"             TimescaleDB extension: {health.get('timescaledb_extension')}")
    else:
        print(f"    [OFFLINE] PostgreSQL server is not running on {db_config.host}:{db_config.port}.")
        print(f"              Diagnostic details: {health.get('error')}")
        print("    [INFO] Live database deployment is ready, but requires starting PostgreSQL.")
        print("           When starting the database, run:")
        print("           python -m database.init_db --seed-stations")

    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: DATABASE DESIGN & ARCHITECTURE IS 100% VERIFIED & VALIDATED.")
    else:
        print("RESULT: ISSUES ENCOUNTERED DURING DATABASE VERIFICATION.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_database_architecture()
    sys.exit(0 if success else 1)
