"""
tests/test_database.py
----------------------
Unit and integration tests for PostgreSQL / TimescaleDB database layer.
- Tests configuration parsing and credential masking.
- Validates database/schema.sql structure, DDL statements, and hypertable definitions.
- Validates health check probe behavior in offline and online modes.
- Validates model representations.
- Marks live database tests with skip conditions when PostgreSQL is unreachable.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from database.config import DatabaseConfig
from database.connection import check_database_health, probe_tcp_port
from database.models import (
    Station,
    RaindropObservation,
    RainfallIntensityRecord,
    PredictionAuditLog,
)
from database.init_db import load_schema_sql, split_sql_statements

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = PROJECT_ROOT / "database" / "schema.sql"


def is_live_db_available() -> bool:
    """Helper to determine whether a live PostgreSQL server is accessible."""
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    ok, _ = probe_tcp_port(host, port, timeout_sec=0.5)
    return ok


class TestDatabaseConfig(unittest.TestCase):
    """Unit tests for database configuration and credential safety."""

    def test_default_configuration_values(self):
        """Check sensible defaults for local development."""
        cfg = DatabaseConfig()
        self.assertEqual(cfg.host, "localhost")
        self.assertEqual(cfg.port, 5432)
        self.assertEqual(cfg.name, "raindrop_db")
        self.assertEqual(cfg.user, "postgres")
        self.assertGreaterEqual(cfg.pool_size, 1)

    def test_safe_url_masks_passwords(self):
        """Verify that get_safe_url() never prints the raw password."""
        cfg = DatabaseConfig(
            host="db.internal.example.com",
            port=5432,
            name="prod_rain",
            user="app_user",
            password="super_secret_password_12345",
        )
        safe_url = cfg.get_safe_url()
        self.assertNotIn("super_secret_password_12345", safe_url)
        self.assertIn("***", safe_url)
        self.assertIn("app_user", safe_url)
        self.assertIn("db.internal.example.com", safe_url)

    def test_sqlalchemy_url_generation(self):
        """Verify proper SQLAlchemy URI format."""
        cfg = DatabaseConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="tester",
            password="pwd",
        )
        raw_url = cfg.get_sqlalchemy_url(driver="psycopg2")
        self.assertEqual(raw_url, "postgresql+psycopg2://tester:pwd@localhost:5432/testdb")


class TestDatabaseSchemaValidation(unittest.TestCase):
    """Static inspection and structural validation of database/schema.sql."""

    def setUp(self):
        self.assertTrue(SCHEMA_FILE.exists(), f"Missing schema at {SCHEMA_FILE}")
        self.schema_sql = load_schema_sql()

    def test_required_tables_present(self):
        """Verify all 4 core tables are defined with IF NOT EXISTS."""
        required_tables = [
            "CREATE TABLE IF NOT EXISTS stations",
            "CREATE TABLE IF NOT EXISTS raindrop_observations",
            "CREATE TABLE IF NOT EXISTS rainfall_intensity_records",
            "CREATE TABLE IF NOT EXISTS prediction_audit_logs",
        ]
        for tbl in required_tables:
            self.assertIn(
                tbl,
                self.schema_sql,
                f"Missing expected table DDL: {tbl}",
            )

    def test_physical_check_constraints(self):
        """Verify scientific boundaries are enforced by schema constraints."""
        # Diameter check
        self.assertIn("diameter_mm > 0.0 AND diameter_mm <= 15.0", self.schema_sql)
        # Velocity check
        self.assertIn("velocity_m_s > 0.0 AND velocity_m_s <= 25.0", self.schema_sql)
        # Volume check
        self.assertIn("volume_mm3 > 0.0", self.schema_sql)

    def test_timescale_hypertable_declaration(self):
        """Verify Timescale hypertable creation logic for time-series partitioning."""
        self.assertIn("create_hypertable", self.schema_sql)
        self.assertIn("'raindrop_observations'", self.schema_sql)
        self.assertIn("'observed_at'", self.schema_sql)
        self.assertIn("INTERVAL '7 days'", self.schema_sql)

    def test_analytical_views_present(self):
        """Verify analytical views are defined for DSD and tail drop analysis."""
        self.assertIn("CREATE OR REPLACE VIEW v_hourly_dsd_summary", self.schema_sql)
        self.assertIn("CREATE OR REPLACE VIEW v_extreme_raindrops", self.schema_sql)

    def test_split_sql_statements_parser(self):
        """Verify parser handles PL/pgSQL dollar quotes without truncating."""
        statements = split_sql_statements(self.schema_sql)
        self.assertGreater(len(statements), 5)
        # Ensure DO $$ blocks are preserved intact
        do_blocks = [s for s in statements if "DO $$" in s]
        self.assertGreaterEqual(len(do_blocks), 2)
        for b in do_blocks:
            self.assertIn("$$;", b)


class TestDatabaseHealthCheck(unittest.TestCase):
    """Unit tests for database connection probe and health monitoring."""

    def test_health_check_offline_behavior(self):
        """
        When PostgreSQL is not running or drivers are missing, check_database_health()
        must return a structured dictionary and never raise an unhandled exception.
        """
        # Test with unreachable port
        dummy_cfg = DatabaseConfig(host="127.0.0.1", port=59999)
        health = check_database_health(dummy_cfg)

        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertIn("is_connected", health)
        self.assertFalse(health["is_connected"])
        self.assertIn("port_listening", health)
        self.assertIn("error", health)
        self.assertIsNotNone(health["error"])


class TestDatabaseModels(unittest.TestCase):
    """Unit tests verifying models/dataclasses definition and instantiation."""

    def test_station_model_instantiation(self):
        """Check station instance attributes."""
        st = Station(
            station_id="ST_01",
            station_name="Cedar Experimental Plot",
            canopy_type="Foliated Cedar",
            instrument_type="Optical Disdrometer",
            latitude=35.5,
            longitude=139.7,
            elevation_m=110.0,
        )
        self.assertEqual(st.station_id, "ST_01")
        self.assertEqual(st.canopy_type, "Foliated Cedar")

    def test_prediction_audit_log_instantiation(self):
        """Check prediction audit log fields."""
        log = PredictionAuditLog(
            input_diameter_mm=1.2,
            input_velocity_m_s=3.4,
            formula_volume_mm3=0.904778,
            xgb_volume_mm3=0.905100,
            latency_ms=1.85,
        )
        self.assertEqual(log.input_diameter_mm, 1.2)
        self.assertEqual(log.xgb_volume_mm3, 0.905100)


class TestOfflineDatabaseOperations(unittest.TestCase):
    """
    Verifies ORM models, foreign keys, relationships, autoincrement, and
    cascade deletion in an isolated in-memory engine without requiring PostgreSQL.
    """

    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base

        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def test_offline_crud_and_relationship_lifecycle(self):
        """Test full insert, retrieve, relationship traversal, and cascade delete."""
        import datetime

        # 1. Insert Station
        station = Station(
            station_id="ST_OAK_01",
            station_name="Oak Canopy Test Station",
            canopy_type="Defoliated Oak",
            latitude=35.69,
            longitude=139.69,
        )
        self.session.add(station)
        self.session.commit()

        # 2. Insert Raindrop Observation
        obs = RaindropObservation(
            observed_at=datetime.datetime(2026, 11, 19, 10, 0, 0),
            station_id="ST_OAK_01",
            diameter_mm=1.40,
            velocity_m_s=4.50,
            volume_mm3=1.4368,
            condition="Defoliated_Oak_Rain",
            source_file="test_raindrop.csv",
        )
        self.session.add(obs)

        # 3. Insert Rainfall Intensity Record
        intensity = RainfallIntensityRecord(
            period_start=datetime.datetime(2026, 11, 19, 10, 0, 0),
            period_end=datetime.datetime(2026, 11, 19, 10, 1, 0),
            station_id="ST_OAK_01",
            rain_rate_mm_h=8.2,
            total_drops=250,
            mean_diameter_mm=1.2,
            mean_velocity_m_s=4.1,
            total_volume_mm3=180.5,
            condition="Defoliated_Oak_Rain",
        )
        self.session.add(intensity)
        self.session.commit()

        # 4. Traversal
        retrieved_st = self.session.query(Station).filter_by(station_id="ST_OAK_01").one()
        self.assertEqual(len(retrieved_st.observations), 1)
        self.assertEqual(len(retrieved_st.intensity_records), 1)
        self.assertEqual(retrieved_st.observations[0].diameter_mm, 1.40)

        # 5. Cascade Delete
        self.session.delete(retrieved_st)
        self.session.commit()
        self.assertEqual(self.session.query(RaindropObservation).count(), 0)
        self.assertEqual(self.session.query(RainfallIntensityRecord).count(), 0)

    def test_prediction_audit_autoincrement(self):
        """Verify autoincrement primary key generation for prediction audit logs."""
        log1 = PredictionAuditLog(
            input_diameter_mm=1.1,
            input_velocity_m_s=3.2,
            formula_volume_mm3=0.6969,
            xgb_volume_mm3=0.6975,
            latency_ms=1.2,
        )
        log2 = PredictionAuditLog(
            input_diameter_mm=2.0,
            input_velocity_m_s=5.5,
            formula_volume_mm3=4.1888,
            xgb_volume_mm3=4.1890,
            latency_ms=1.5,
        )
        self.session.add_all([log1, log2])
        self.session.commit()

        self.assertIsNotNone(log1.request_id)
        self.assertIsNotNone(log2.request_id)
        self.assertEqual(log2.request_id, log1.request_id + 1)


class TestLiveDatabaseIntegration(unittest.TestCase):
    """
    Integration tests requiring a live PostgreSQL / TimescaleDB instance.
    Automatically skipped if database server is not accessible on network.
    """

    @unittest.skipIf(not is_live_db_available(), "PostgreSQL/TimescaleDB live server is not accessible on localhost:5432.")
    def test_live_database_connection_and_version(self):
        """Verify actual query execution on live PostgreSQL instance."""
        health = check_database_health()
        self.assertTrue(health["is_connected"], f"Health check failed: {health.get('error')}")
        self.assertIsNotNone(health["latency_ms"])
        self.assertGreater(health["latency_ms"], 0.0)

    @unittest.skipIf(not is_live_db_available(), "PostgreSQL/TimescaleDB live server is not accessible on localhost:5432.")
    def test_live_stations_seeded(self):
        """Verify baseline monitoring stations exist in live raindrop_db."""
        from database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT station_id FROM stations ORDER BY station_id")).fetchall()
            station_ids = [r[0] for r in result]
            self.assertIn("RD1_Foliated_Cedar", station_ids)
            self.assertIn("RD2_Defoliated_Oak", station_ids)
            self.assertIn("RD3_Open_Field", station_ids)

    @unittest.skipIf(not is_live_db_available(), "PostgreSQL/TimescaleDB live server is not accessible on localhost:5432.")
    def test_live_crud_and_cascade_lifecycle(self):
        """Perform real CRUD lifecycle and verify cascade delete in live PostgreSQL."""
        import datetime
        from database.connection import get_session_factory
        factory = get_session_factory()
        self.assertIsNotNone(factory)

        test_st_id = "TEMP_TEST_PG19_VERIFY"
        with factory() as session:
            try:
                # Cleanup any residue
                existing = session.query(Station).filter_by(station_id=test_st_id).first()
                if existing:
                    session.delete(existing)
                    session.commit()

                # 1. Create Station
                st = Station(
                    station_id=test_st_id,
                    station_name="Temporary Verification Station",
                    canopy_type="Live Test Canopy",
                    instrument_type="Optical Disdrometer",
                    latitude=35.123,
                    longitude=139.456,
                    elevation_m=100.0,
                )
                session.add(st)
                session.commit()

                # 2. Insert Observation
                now = datetime.datetime.now()
                obs = RaindropObservation(
                    observed_at=now,
                    station_id=test_st_id,
                    diameter_mm=2.50,
                    velocity_m_s=6.20,
                    volume_mm3=8.1812,
                    condition="Live_Test_Rain",
                    source_file="live_verification.csv",
                )
                session.add(obs)
                session.commit()

                # 3. Read & Verify
                queried_obs = session.query(RaindropObservation).filter_by(station_id=test_st_id).first()
                self.assertIsNotNone(queried_obs)
                self.assertEqual(queried_obs.diameter_mm, 2.50)
                self.assertIsNotNone(queried_obs.observation_id)

                # 4. Update
                queried_obs.velocity_m_s = 6.25
                session.commit()
                queried_obs = session.query(RaindropObservation).filter_by(station_id=test_st_id).first()
                self.assertEqual(queried_obs.velocity_m_s, 6.25)

                # 5. Cascade Delete
                session.delete(st)
                session.commit()

                remaining_obs = session.query(RaindropObservation).filter_by(station_id=test_st_id).count()
                self.assertEqual(remaining_obs, 0, "Cascade deletion did not remove observations!")

            finally:
                rem = session.query(Station).filter_by(station_id=test_st_id).first()
                if rem:
                    session.delete(rem)
                    session.commit()

    @unittest.skipIf(not is_live_db_available(), "PostgreSQL/TimescaleDB live server is not accessible on localhost:5432.")
    def test_live_timescaledb_detection(self):
        """
        Confirm that check_live_timescaledb_status accurately reports
        TimescaleDB as NOT_INSTALLED on the live PostgreSQL 19 Windows instance,
        and raindrop_observations as a regular relational table.
        """
        from database.connection import get_engine
        from scripts.verify_database_setup import check_live_timescaledb_status

        engine = get_engine()
        with engine.connect() as conn:
            ts_status = check_live_timescaledb_status(conn)
            self.assertFalse(ts_status["installed"])
            self.assertEqual(ts_status["status"], "NOT_INSTALLED")
            self.assertFalse(ts_status["is_hypertable"])
            self.assertIsNone(ts_status["chunk_interval"])


class TestTimescaleDBVerification(unittest.TestCase):
    """
    Unit tests for live TimescaleDB hypertable and chunk interval status detection.
    Covers both available and unavailable scenarios using mocked database connections.
    """

    def test_timescaledb_not_installed_scenario(self):
        """
        When TimescaleDB is not installed in PostgreSQL, check_live_timescaledb_status
        must report status 'NOT_INSTALLED', is_hypertable=False, and not fail.
        """
        from scripts.verify_database_setup import check_live_timescaledb_status
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None

        res = check_live_timescaledb_status(mock_conn)
        self.assertFalse(res["installed"])
        self.assertIsNone(res["version"])
        self.assertFalse(res["is_hypertable"])
        self.assertIsNone(res["chunk_interval"])
        self.assertFalse(res["chunk_interval_matches"])
        self.assertEqual(res["status"], "NOT_INSTALLED")
        self.assertIn("standard PostgreSQL", res["message"])

    def test_timescaledb_installed_and_hypertable_verified_scenario(self):
        """
        When TimescaleDB is installed and raindrop_observations is a hypertable
        with a 7-day chunk interval, check_live_timescaledb_status must report 'VERIFIED'.
        """
        from scripts.verify_database_setup import check_live_timescaledb_status
        from unittest.mock import MagicMock

        mock_conn = MagicMock()

        def mock_execute(query):
            query_str = str(query)
            mock_res = MagicMock()
            if "pg_extension" in query_str:
                mock_res.fetchone.return_value = ("2.14.2",)
            elif "timescaledb_information.hypertables" in query_str:
                mock_res.fetchone.return_value = ("raindrop_observations",)
            elif "timescaledb_information.dimensions" in query_str:
                mock_res.fetchone.return_value = ("7 days",)
            else:
                mock_res.fetchone.return_value = None
            return mock_res

        mock_conn.execute.side_effect = mock_execute

        res = check_live_timescaledb_status(mock_conn)
        self.assertTrue(res["installed"])
        self.assertEqual(res["version"], "2.14.2")
        self.assertTrue(res["is_hypertable"])
        self.assertEqual(res["chunk_interval"], "7 days")
        self.assertTrue(res["chunk_interval_matches"])
        self.assertEqual(res["status"], "VERIFIED")
        self.assertIn("7-day chunk interval", res["message"])

    def test_timescaledb_installed_but_table_is_regular_relational(self):
        """
        When TimescaleDB is installed but raindrop_observations was not converted
        into a hypertable, report 'NOT_HYPERTABLE'.
        """
        from scripts.verify_database_setup import check_live_timescaledb_status
        from unittest.mock import MagicMock

        mock_conn = MagicMock()

        def mock_execute(query):
            query_str = str(query)
            mock_res = MagicMock()
            if "pg_extension" in query_str:
                mock_res.fetchone.return_value = ("2.14.2",)
            elif "hypertables" in query_str or "_timescaledb_catalog" in query_str:
                mock_res.fetchone.return_value = None
            else:
                mock_res.fetchone.return_value = None
            return mock_res

        mock_conn.execute.side_effect = mock_execute

        res = check_live_timescaledb_status(mock_conn)
        self.assertTrue(res["installed"])
        self.assertFalse(res["is_hypertable"])
        self.assertEqual(res["status"], "NOT_HYPERTABLE")
        self.assertIn("not configured as a hypertable", res["message"])

    def test_timescaledb_installed_hypertable_chunk_interval_mismatch(self):
        """
        When hypertable chunk interval differs from 7 days, report 'INTERVAL_MISMATCH'.
        """
        from scripts.verify_database_setup import check_live_timescaledb_status
        from unittest.mock import MagicMock

        mock_conn = MagicMock()

        def mock_execute(query):
            query_str = str(query)
            mock_res = MagicMock()
            if "pg_extension" in query_str:
                mock_res.fetchone.return_value = ("2.14.2",)
            elif "timescaledb_information.hypertables" in query_str:
                mock_res.fetchone.return_value = ("raindrop_observations",)
            elif "timescaledb_information.dimensions" in query_str:
                mock_res.fetchone.return_value = ("1 day",)
            else:
                mock_res.fetchone.return_value = None
            return mock_res

        mock_conn.execute.side_effect = mock_execute

        res = check_live_timescaledb_status(mock_conn)
        self.assertTrue(res["installed"])
        self.assertTrue(res["is_hypertable"])
        self.assertFalse(res["chunk_interval_matches"])
        self.assertEqual(res["status"], "INTERVAL_MISMATCH")
        self.assertEqual(res["chunk_interval"], "1 day")


if __name__ == "__main__":
    unittest.main()
