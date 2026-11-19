"""
database/connection.py
----------------------
Database engine and connection pool management for PostgreSQL / TimescaleDB.
Provides:
- SQLAlchemy engine with connection pooling and pre-ping validation.
- FastAPI dependency `get_db()` for safe request-scoped session lifecycle.
- Safe `check_database_health()` probe measuring connection latency and extension state.
- Graceful connection cleanup on server shutdown.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any, Dict, Generator, Optional, Tuple

from database.config import db_config, DatabaseConfig

logger = logging.getLogger("database.connection")

# ── Optional driver imports with graceful fallbacks ──
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import declarative_base, sessionmaker, Session
    _HAS_SQLALCHEMY = True
except ImportError:
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    declarative_base = None  # type: ignore[assignment]
    sessionmaker = None  # type: ignore[assignment]
    Session = Any  # type: ignore[assignment,misc]
    _HAS_SQLALCHEMY = False

try:
    import psycopg2
    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    _HAS_PSYCOPG2 = False

try:
    import psycopg
    _HAS_PSYCOPG3 = True
except ImportError:
    psycopg = None  # type: ignore[assignment]
    _HAS_PSYCOPG3 = False

# Global singletons
_ENGINE = None
_SESSION_LOCAL = None


def get_driver_name() -> Optional[str]:
    """Detect available PostgreSQL driver."""
    if _HAS_PSYCOPG2:
        return "psycopg2"
    if _HAS_PSYCOPG3:
        return "psycopg"
    return None


def get_engine(cfg: Optional[DatabaseConfig] = None):
    """
    Get or create SQLAlchemy engine singleton with connection pooling.
    Uses pool_pre_ping to drop dead connections automatically.
    """
    global _ENGINE
    if not _HAS_SQLALCHEMY:
        return None

    if _ENGINE is None:
        cfg = cfg or db_config
        driver = get_driver_name()
        # Fall back to default driver if none detected
        driver_str = driver if driver else "psycopg2"
        url = cfg.get_sqlalchemy_url(driver=driver_str)
        try:
            _ENGINE = create_engine(
                url,
                pool_size=cfg.pool_size,
                max_overflow=cfg.max_overflow,
                pool_timeout=cfg.pool_timeout,
                pool_recycle=cfg.pool_recycle,
                pool_pre_ping=True,  # Test connections before checkout
                echo=False,
            )
            logger.info("Database engine initialized: %s", cfg.get_safe_url())
        except Exception as exc:
            logger.warning("Failed to initialize database engine: %s", exc)
            return None

    return _ENGINE


def get_session_factory(cfg: Optional[DatabaseConfig] = None):
    """Get or create sessionmaker singleton."""
    global _SESSION_LOCAL
    if not _HAS_SQLALCHEMY:
        return None

    if _SESSION_LOCAL is None:
        engine = get_engine(cfg)
        if engine is not None:
            _SESSION_LOCAL = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine,
            )
    return _SESSION_LOCAL


def get_db() -> Generator[Any, None, None]:
    """
    FastAPI dependency yielding a request-scoped database session.
    Ensures connection is returned to pool even if an exception occurs.
    """
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError(
            "SQLAlchemy / database driver is not installed or engine could not be created."
        )

    session = factory()
    try:
        yield session
    finally:
        session.close()


def probe_tcp_port(host: str, port: int, timeout_sec: float = 2.0) -> Tuple[bool, Optional[str]]:
    """Safe socket-level check to determine if database port is listening."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    try:
        result = sock.connect_ex((host, port))
        if result == 0:
            return True, None
        return False, f"TCP connect to ({host}:{port}) failed with errno {result}."
    except Exception as exc:
        return False, str(exc)
    finally:
        sock.close()


def check_database_health(cfg: Optional[DatabaseConfig] = None) -> Dict[str, Any]:
    """
    Comprehensive health check for PostgreSQL / TimescaleDB.
    Validates:
    1. Driver availability (SQLAlchemy + psycopg2 / psycopg).
    2. TCP port reachability (socket probe).
    3. Live database query execution ('SELECT 1;').
    4. Query roundtrip latency in milliseconds.
    5. TimescaleDB extension presence and installed version.
    """
    cfg = cfg or db_config
    base_info = {
        "database": "postgresql",
        "host": cfg.host,
        "port": cfg.port,
        "database_name": cfg.name,
        "safe_url": cfg.get_safe_url(),
        "timestamp": time.time(),
    }

    # Step 1: Check driver availability
    driver = get_driver_name()
    if not _HAS_SQLALCHEMY or not driver:
        # Check TCP port anyway to give precise diagnosis
        tcp_ok, tcp_err = probe_tcp_port(cfg.host, cfg.port, timeout_sec=1.0)
        return {
            **base_info,
            "status": "driver_missing",
            "is_connected": False,
            "latency_ms": None,
            "timescaledb_extension": None,
            "timescaledb_version": None,
            "port_listening": tcp_ok,
            "error": (
                "Database driver missing. Install 'sqlalchemy' and 'psycopg2-binary' (or 'psycopg'). "
                + (f"Port {cfg.port} is reachable." if tcp_ok else f"Port {cfg.port} is not listening.")
            ),
        }

    # Step 2: Probe TCP port reachability
    tcp_ok, tcp_err = probe_tcp_port(cfg.host, cfg.port, timeout_sec=2.0)
    if not tcp_ok:
        return {
            **base_info,
            "status": "unreachable",
            "is_connected": False,
            "latency_ms": None,
            "timescaledb_extension": None,
            "timescaledb_version": None,
            "port_listening": False,
            "error": f"PostgreSQL server not reachable at {cfg.host}:{cfg.port}. ({tcp_err})",
        }

    # Step 3: Execute live query via SQLAlchemy
    engine = get_engine(cfg)
    if engine is None:
        return {
            **base_info,
            "status": "initialization_failed",
            "is_connected": False,
            "latency_ms": None,
            "timescaledb_extension": None,
            "timescaledb_version": None,
            "port_listening": True,
            "error": "Failed to create SQLAlchemy engine with given credentials.",
        }

    t_start = time.perf_counter()
    try:
        with engine.connect() as conn:
            # Ping query
            conn.execute(text("SELECT 1;"))
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            # Check TimescaleDB extension
            res = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            ).fetchone()

            has_timescale = res is not None
            timescale_ver = str(res[0]) if res else None

            return {
                **base_info,
                "status": "healthy",
                "is_connected": True,
                "latency_ms": latency_ms,
                "timescaledb_extension": has_timescale,
                "timescaledb_version": timescale_ver,
                "port_listening": True,
                "error": None,
            }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return {
            **base_info,
            "status": "connection_error",
            "is_connected": False,
            "latency_ms": latency_ms,
            "timescaledb_extension": None,
            "timescaledb_version": None,
            "port_listening": True,
            "error": f"Database query failed: {type(exc).__name__}: {str(exc)}",
        }


def log_prediction_audit(
    input_diameter: float,
    input_velocity: float,
    formula_vol: float,
    xgb_vol: Optional[float],
    latency_ms: float,
    client_ip: Optional[str] = None,
) -> Tuple[Optional[int], str]:
    """
    Safely persist a prediction audit record to prediction_audit_logs if database is available.
    Returns (request_id, audit_status).

    Guarantees:
    - Never raises an unhandled exception if database is unreachable.
    - Never pretends a record was saved when it was not.
    - Returns (None, 'skipped (database unavailable)') when database is offline.
    """
    driver = get_driver_name()
    if not _HAS_SQLALCHEMY or not driver:
        return None, "skipped (database unavailable)"

    # Fast socket check before attempting engine checkout
    tcp_ok, _ = probe_tcp_port(db_config.host, db_config.port, timeout_sec=0.2)
    if not tcp_ok:
        return None, "skipped (database unavailable)"

    factory = get_session_factory()
    if factory is None:
        return None, "skipped (database unavailable)"

    try:
        from database.models import PredictionAuditLog
        with factory() as session:
            record = PredictionAuditLog(
                input_diameter_mm=input_diameter,
                input_velocity_m_s=input_velocity,
                formula_volume_mm3=formula_vol,
                xgb_volume_mm3=xgb_vol,
                latency_ms=latency_ms,
                client_ip=client_ip,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.request_id, "persisted"
    except Exception as exc:
        logger.warning("Database audit logging failed: %s", exc)
        return None, "skipped (database unavailable)"


def close_database_pool() -> None:
    """Safely dispose the SQLAlchemy engine and close all idle pooled connections."""
    global _ENGINE, _SESSION_LOCAL
    if _ENGINE is not None:
        try:
            _ENGINE.dispose()
            logger.info("Database connection pool disposed successfully.")
        except Exception as exc:
            logger.warning("Error disposing database pool: %s", exc)
        finally:
            _ENGINE = None
            _SESSION_LOCAL = None
