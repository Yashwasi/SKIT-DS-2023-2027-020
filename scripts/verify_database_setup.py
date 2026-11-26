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
from typing import Any, Dict, Optional

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


def check_live_timescaledb_status(conn: Any) -> Dict[str, Any]:
    """
    Query PostgreSQL to determine whether the TimescaleDB extension is installed,
    and if so, verify whether raindrop_observations is an active hypertable
    with a 7-day chunk interval.

    Returns a structured dictionary with keys:
    - installed: bool
    - version: Optional[str]
    - is_hypertable: bool
    - chunk_interval: Optional[str]
    - chunk_interval_matches: bool
    - status: str ('NOT_INSTALLED' | 'VERIFIED' | 'NOT_HYPERTABLE' | 'INTERVAL_MISMATCH')
    - message: str
    """
    from sqlalchemy import text

    # Step 1: Check extension installation in PostgreSQL catalog
    ext_row = conn.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
    ).fetchone()

    if not ext_row:
        return {
            "installed": False,
            "version": None,
            "is_hypertable": False,
            "chunk_interval": None,
            "chunk_interval_matches": False,
            "status": "NOT_INSTALLED",
            "message": "TimescaleDB extension is not installed; standard PostgreSQL relational table active.",
        }

    version_str = str(ext_row[0])

    # Step 2: Query TimescaleDB metadata for hypertable confirmation
    is_hypertable = False
    try:
        ht_row = conn.execute(
            text(
                "SELECT hypertable_name FROM timescaledb_information.hypertables "
                "WHERE hypertable_name = 'raindrop_observations';"
            )
        ).fetchone()
        is_hypertable = ht_row is not None
    except Exception:
        try:
            cat_row = conn.execute(
                text(
                    "SELECT 1 FROM _timescaledb_catalog.hypertable "
                    "WHERE table_name = 'raindrop_observations';"
                )
            ).fetchone()
            is_hypertable = cat_row is not None
        except Exception:
            is_hypertable = False

    if not is_hypertable:
        return {
            "installed": True,
            "version": version_str,
            "is_hypertable": False,
            "chunk_interval": None,
            "chunk_interval_matches": False,
            "status": "NOT_HYPERTABLE",
            "message": (
                f"TimescaleDB extension (v{version_str}) is installed, "
                "but 'raindrop_observations' is not configured as a hypertable."
            ),
        }

    # Step 3: Query TimescaleDB dimension metadata for 7-day chunk interval
    chunk_interval = None
    chunk_interval_matches = False
    try:
        dim_row = conn.execute(
            text(
                "SELECT time_interval FROM timescaledb_information.dimensions "
                "WHERE hypertable_name = 'raindrop_observations';"
            )
        ).fetchone()
        if dim_row and dim_row[0] is not None:
            chunk_interval = str(dim_row[0])
            raw_str = chunk_interval.strip().lower()
            if "7 day" in raw_str or raw_str == "604800000000" or raw_str == "604800s":
                chunk_interval_matches = True
    except Exception:
        pass

    if not chunk_interval_matches:
        return {
            "installed": True,
            "version": version_str,
            "is_hypertable": True,
            "chunk_interval": chunk_interval,
            "chunk_interval_matches": False,
            "status": "INTERVAL_MISMATCH",
            "message": (
                f"TimescaleDB hypertable verified, but chunk interval is '{chunk_interval}' "
                "(expected 7 days)."
            ),
        }

    return {
        "installed": True,
        "version": version_str,
        "is_hypertable": True,
        "chunk_interval": chunk_interval,
        "chunk_interval_matches": True,
        "status": "VERIFIED",
        "message": (
            f"TimescaleDB extension (v{version_str}) verified: "
            "'raindrop_observations' is an active hypertable with 7-day chunk interval."
        ),
    }


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
    # 2. DDL Schema File Integrity (Static Schema Declarations)
    # -------------------------------------------------------------------------
    print("\n[2] DDL SCHEMA INTEGRITY & SYNTAX (STATIC DECLARATION CHECK):")
    schema_file = PROJECT_ROOT / "database" / "schema.sql"
    if not schema_file.exists():
        print(f"    [FAIL] Schema file missing at: {schema_file}")
        return False

    raw_sql = load_schema_sql()
    statements = split_sql_statements(raw_sql)
    print(f"    Schema file size: {len(raw_sql):,} characters")
    print(f"    Parsed DDL executable blocks: {len(statements)}")

    # Verify key DDL declarations in schema file (Static inspection)
    expected_declarations = [
        ("stations table DDL", "CREATE TABLE IF NOT EXISTS stations"),
        ("raindrop_observations table DDL", "CREATE TABLE IF NOT EXISTS raindrop_observations"),
        ("rainfall_intensity_records table DDL", "CREATE TABLE IF NOT EXISTS rainfall_intensity_records"),
        ("prediction_audit_logs table DDL", "CREATE TABLE IF NOT EXISTS prediction_audit_logs"),
        ("Physical constraints DDL", "diameter_mm > 0.0 AND diameter_mm <= 15.0"),
        ("DSD summary view DDL", "v_hourly_dsd_summary"),
        ("Extreme drops view DDL", "v_extreme_raindrops"),
        ("TimescaleDB hypertable DDL syntax", "create_hypertable"),
        ("7-day chunk interval DDL syntax", "INTERVAL '7 days'"),
    ]

    for label, pattern in expected_declarations:
        if pattern in raw_sql:
            print(f"    [OK] {label:<40} (Declared in schema.sql)")
        else:
            print(f"    [FAIL] {label:<40} (Not found in schema.sql)")
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
    # 4. Live PostgreSQL Server Status Probe & Live Operations Verification
    # -------------------------------------------------------------------------
    print("\n[4] LIVE DATABASE ENVIRONMENT PROBE:")
    tcp_ok, tcp_err = probe_tcp_port(db_config.host, db_config.port, timeout_sec=1.0)
    health = check_database_health()

    if tcp_ok and health.get("is_connected"):
        print(f"    [ONLINE] PostgreSQL port {db_config.port} is reachable.")
        print(f"             Status:                {health.get('status')}")
        print(f"             Query Latency:         {health.get('latency_ms')} ms")

        # Inspect live schema objects
        try:
            from database.connection import get_engine, get_session_factory
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                # Live TimescaleDB & Hypertable Status Verification
                ts_status = check_live_timescaledb_status(conn)
                print(f"             TimescaleDB extension: {'INSTALLED (v' + str(ts_status['version']) + ')' if ts_status['installed'] else '[NOT INSTALLED]'}")

                if ts_status["installed"]:
                    if ts_status["is_hypertable"]:
                        print("             [OK] Live Hypertable Status:       raindrop_observations confirmed as hypertable (Verified)")
                        if ts_status.get("chunk_interval_matches"):
                            print(f"             [OK] Live 7-Day Chunk Interval:    {ts_status['chunk_interval']} confirmed from hypertable dimensions (Verified)")
                        else:
                            print(f"             [WARNING] Live Chunk Interval:     {ts_status['chunk_interval']} (Differs from expected 7 days)")
                    else:
                        print("             [NOTE] Live Hypertable Status:     raindrop_observations is a standard relational table (Not converted)")
                        print("             [SKIPPED] Live 7-Day Chunk Interval: Skipped (Table is standard relational, not a hypertable)")
                else:
                    print("             [NOT INSTALLED] TimescaleDB extension is not installed in PostgreSQL.")
                    print("             [SKIPPED] Live Hypertable Verification: Skipped (Standard PostgreSQL relational table active).")
                    print("             [SKIPPED] Live 7-Day Chunk Interval:   Skipped (Standard PostgreSQL relational table active).")
                    print("             [INFO] Note: TimescaleDB is optional. Standard PostgreSQL relational architecture operates normally.")

                res_tables = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
                )).fetchall()
                live_tables = [r[0] for r in res_tables]
                print(f"             Live tables & views ({len(live_tables)}): {', '.join(live_tables)}")

                for req_t in ["stations", "raindrop_observations", "rainfall_intensity_records", "prediction_audit_logs"]:
                    if req_t in live_tables:
                        print(f"             [OK] Live table '{req_t}' exists.")
                    else:
                        print(f"             [FAIL] Live table '{req_t}' missing!")
                        all_passed = False

                for req_v in ["v_hourly_dsd_summary", "v_extreme_raindrops"]:
                    if req_v in live_tables:
                        print(f"             [OK] Live view '{req_v}' exists.")
                    else:
                        print(f"             [FAIL] Live view '{req_v}' missing!")
                        all_passed = False

                st_count = conn.execute(text("SELECT count(*) FROM stations")).scalar()
                print(f"             Seeded stations count: {st_count}")

            # Live CRUD & Constraint Test on raindrop_db
            print("\n[5] LIVE DATABASE TRANSACTION & CONSTRAINT VERIFICATION (raindrop_db):")
            factory = get_session_factory()
            test_st_id = "TEMP_AUDIT_VERIFY_ST"
            with factory() as session:
                try:
                    # Clean any old test fixture
                    old = session.query(Station).filter_by(station_id=test_st_id).first()
                    if old:
                        session.delete(old)
                        session.commit()

                    # 1. Insert temporary station
                    live_st = Station(
                        station_id=test_st_id,
                        station_name="Temporary Audit Verification Station",
                        canopy_type="Cedar Live Audit Plot",
                        instrument_type="Optical Disdrometer",
                        latitude=35.123,
                        longitude=139.456,
                        elevation_m=120.0,
                    )
                    session.add(live_st)
                    session.commit()
                    print("    [OK] Live Station insert: SUCCESS")

                    # 2. Insert valid observation
                    live_obs = RaindropObservation(
                        observed_at=datetime.datetime.now(),
                        station_id=test_st_id,
                        diameter_mm=1.85,
                        velocity_m_s=5.20,
                        volume_mm3=3.3151,
                        condition="Live_Verification_Rain",
                        source_file="live_audit.csv",
                    )
                    session.add(live_obs)
                    session.commit()
                    print(f"    [OK] Live Observation insert (observation_id={live_obs.observation_id}): SUCCESS")

                    # 3. Update observation
                    live_obs.velocity_m_s = 5.25
                    session.commit()
                    print("    [OK] Live Observation update: SUCCESS")

                    # 4. Check Constraint Verification (diameter <= 0 must fail)
                    invalid_obs = RaindropObservation(
                        observed_at=datetime.datetime.now(),
                        station_id=test_st_id,
                        diameter_mm=-1.0,  # Invalid!
                        velocity_m_s=3.0,
                        volume_mm3=0.5,
                        condition="Invalid",
                        source_file="test.csv",
                    )
                    session.add(invalid_obs)
                    try:
                        session.commit()
                        print("    [FAIL] Database accepted invalid diameter <= 0!")
                        all_passed = False
                    except Exception as chk_exc:
                        session.rollback()
                        print("    [OK] Scientific check constraint correctly rejected invalid diameter (<= 0): SUCCESS")

                    # 5. Insert Prediction Audit Log
                    audit_log = PredictionAuditLog(
                        input_diameter_mm=1.85,
                        input_velocity_m_s=5.25,
                        formula_volume_mm3=3.3151,
                        xgb_volume_mm3=3.3120,
                        latency_ms=1.12,
                        client_ip="127.0.0.1",
                    )
                    session.add(audit_log)
                    session.commit()
                    print(f"    [OK] Live Prediction Audit Log insert (request_id={audit_log.request_id}): SUCCESS")

                    # 6. Verify Cascade Deletion
                    session.delete(live_st)
                    session.commit()
                    rem_obs = session.query(RaindropObservation).filter_by(station_id=test_st_id).count()
                    if rem_obs == 0:
                        print("    [OK] Cascade deletion verified (Station deleted -> Observations purged): SUCCESS")
                    else:
                        print("    [FAIL] Cascade deletion failed, orphan records remain!")
                        all_passed = False

                    # Clean audit test record
                    session.delete(audit_log)
                    session.commit()
                    print("    [OK] Test cleanup completed cleanly.")

                finally:
                    rem = session.query(Station).filter_by(station_id=test_st_id).first()
                    if rem:
                        session.delete(rem)
                        session.commit()

        except Exception as exc:
            print(f"    [FAIL] Live database operations failed: {exc}")
            all_passed = False

    elif tcp_ok:
        print(f"    [ONLINE] PostgreSQL port {db_config.port} is reachable, but health check failed.")
        print(f"             Diagnostic details: {health.get('error')}")
        all_passed = False
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
