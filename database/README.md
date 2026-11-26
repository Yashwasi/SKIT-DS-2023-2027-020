# Rain Drop Volume Prediction Using AI — Database Architecture

**Project ID:** SKIT/DS/2023-2027/F-090  
**Domain:** Machine Learning, Environmental Data Science & Database Management  
**Engine:** PostgreSQL with TimescaleDB Extension  
**Backend:** FastAPI REST APIs  

---

## 1. Architecture Overview

The database layer manages both high-frequency disdrometer time-series records and REST API inference audit logs:

```
                                    +-----------------------+
                                    |     Optical Data      |
                                    |  (Disdrometer CSVs)   |
                                    +-----------+-----------+
                                                |
                                                v
+-------------------+              +------------+-----------+
|    FastAPI App    |              | database/init_db.py    |
|   (api/main.py)   |              | (Safe schema deploy)   |
+---------+---------+              +------------+-----------+
          |                                     |
          | /health/database                    | DDL / Initial Seed
          v                                     v
+---------+-------------------------------------+-----------+
|                  PostgreSQL 16 / TimescaleDB              |
|                                                           |
|  - stations (Relational metadata table)                   |
|  - raindrop_observations (Timescale hypertable, 7-day)    |
|  - rainfall_intensity_records (Aggregated rain rates)     |
|  - prediction_audit_logs (REST API inference history)     |
|  - v_hourly_dsd_summary (Drop Size Distribution View)     |
|  - v_extreme_raindrops (Tail drops > 2.6 mm View)         |
+-----------------------------------------------------------+
```

### Key Architectural Decisions
1. **TimescaleDB Hypertable:** `raindrop_observations` stores high-frequency individual particle records. It is partitioned on `observed_at` using **7-day chunks** to optimize query performance and time-range data pruning.
2. **Conditional DDL Fallback:** Schema statements use conditional `DO $$` PL/pgSQL blocks to safely deploy on standard vanilla PostgreSQL instances even if the TimescaleDB extension is not pre-installed.
3. **Connection Pooling & Pre-Ping:** `database/connection.py` configures a pooled SQLAlchemy engine with `pool_pre_ping=True` to detect and refresh stale TCP connections automatically.
4. **Credential Security:** All database credentials are read exclusively from environment variables (`.env`). The helper `DatabaseConfig.get_safe_url()` masks passwords (e.g. `postgresql://user:***@host:5432/db`) to prevent credential leakage in logs.

---

## 2. File Organization

| File | Purpose |
|------|---------|
| `database/schema.sql` | Production PostgreSQL / TimescaleDB DDL (tables, hypertable declarations, indexes, check constraints, views). |
| `database/config.py` | Environment variable loader, configuration dataclass, and safe URL masking (with special character quote handling). |
| `database/connection.py` | SQLAlchemy engine, connection pooling, FastAPI `get_db()` dependency, `log_prediction_audit()`, and `check_database_health()`. |
| `database/models.py` | SQLAlchemy ORM models (`Station`, `RaindropObservation`, `RainfallIntensityRecord`, `PredictionAuditLog`) with dataclass fallbacks. |
| `database/init_db.py` | CLI script for safe schema deployment (`--dry-run`, `--seed-stations`). |
| `.env.example` | Template for database environment variables and pool settings. |
| `tests/test_database.py` | Unit and integration tests covering configuration, schema validation, health checks, and live connections. |
| `scripts/verify_database_setup.py` | Automated pre-flight and live verification script executing real CRUD and check constraint assertions against `raindrop_db`. |

---

## 3. Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your PostgreSQL server credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=raindrop_db
   DB_USER=postgres
   DB_PASSWORD=your_secure_password
   DB_SSLMODE=prefer

   DB_POOL_SIZE=5
   DB_MAX_OVERFLOW=10
   DB_POOL_TIMEOUT=30
   DB_POOL_RECYCLE=1800
   ```

*(Note: `.env` is ignored by Git in `.gitignore` to prevent secret exposure).*

---

## 4. Starting PostgreSQL & TimescaleDB Compatibility

### Native Windows PostgreSQL 19 (Current Verified Setup)
The project is verified against PostgreSQL 19 running locally via the Windows Service `postgresql-x64-19` on port 5432.
```powershell
Get-Service postgresql-x64-19
```

> **Note on TimescaleDB Compatibility with PostgreSQL 19:**  
> As of late 2026, the official TimescaleDB Windows extension supports PostgreSQL versions 14 through 17. Because this host runs PostgreSQL 19, TimescaleDB is not yet available for PG 19 on Windows.  
> The project architecture handles this gracefully via conditional PL/pgSQL DDL blocks (`DO $$ ... $$;` in `database/schema.sql`). On PostgreSQL 19, standard PostgreSQL relational tables, composite indexing, check constraints, foreign key cascades, and analytical views run with 100% operational fidelity (sub-millisecond latency).

### Optional: TimescaleDB via Docker (PostgreSQL 16)
If strict TimescaleDB hypertable chunking is required in development or production:
```bash
docker run -d \
  --name timescale \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=your_secure_password \
  -e POSTGRES_DB=raindrop_db \
  timescale/timescaledb:latest-pg16
```

---

## 5. Schema Deployment & Initialization

1. **Dry-Run (Pre-flight syntax and statement inspection):**
   ```bash
   python -m database.init_db --dry-run
   ```
2. **Execute Schema Deployment (Safe non-destructive):**
   ```bash
   python -m database.init_db --seed-stations
   ```

*Guarantees: Never drops or truncates existing tables or databases. Uses `IF NOT EXISTS` on all objects.*

---

## 6. FastAPI Backend Integration

### Database Health Check Endpoint
The backend exposes `GET /health/database`:
- **When online & connected:** Returns `HTTP 200 OK` with roundtrip query latency in ms and TimescaleDB extension version.
- **When offline / unreachable:** Returns `HTTP 503 Service Unavailable` with structured diagnostic information (masking passwords) suitable for cloud probes.

Example Response (Connected):
```json
{
  "database": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database_name": "raindrop_db",
  "safe_url": "postgresql://postgres:***@localhost:5432/raindrop_db",
  "status": "healthy",
  "is_connected": true,
  "latency_ms": 2.45,
  "timescaledb_extension": true,
  "timescaledb_version": "2.14.2",
  "port_listening": true,
  "error": null
}
```

---

## 7. Running Tests

Run database unit and integration tests:
```bash
python -m unittest tests/test_database.py
```

Run entire project test suite:
```bash
python -m unittest discover tests
```
*(Tests requiring a live database are decorated with `@unittest.skipIf(...)` and automatically skip when PostgreSQL is offline).*
