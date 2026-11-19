-- ============================================================================
-- Rain Drop Volume Prediction Using AI
-- Database Architecture Schema (PostgreSQL / TimescaleDB)
--
-- Session: 2026-27 | Project ID: SKIT/DS/2023-2027/F-090
-- Form-2 User Story: Connect database with backend (01/11/2026 - 30/11/2026)
--                    "Design project database"
-- Author: Shubham Saini (Data Engineering & Database Management)
-- ============================================================================

-- Enable TimescaleDB extension if available in the PostgreSQL instance
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'
    ) THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
        RAISE NOTICE 'TimescaleDB extension enabled successfully.';
    ELSE
        RAISE NOTICE 'TimescaleDB extension not present on this server. Proceeding with standard PostgreSQL relational tables.';
    END IF;
END
$$;

-- ----------------------------------------------------------------------------
-- 1. Stations & Environmental Conditions Table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stations (
    station_id          VARCHAR(64) PRIMARY KEY,
    station_name        VARCHAR(128) NOT NULL,
    canopy_type         VARCHAR(64) NOT NULL,       -- e.g. 'Foliated Cedar', 'Open Field'
    instrument_type     VARCHAR(64) DEFAULT 'Optical Disdrometer',
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    elevation_m         DOUBLE PRECISION,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 2. Individual Raindrop Observations (High-Volume Time-Series Table)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raindrop_observations (
    observed_at         TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    station_id          VARCHAR(64) NOT NULL,
    diameter_mm         DOUBLE PRECISION NOT NULL CHECK (diameter_mm > 0.0 AND diameter_mm <= 15.0),
    velocity_m_s        DOUBLE PRECISION NOT NULL CHECK (velocity_m_s > 0.0 AND velocity_m_s <= 25.0),
    volume_mm3          DOUBLE PRECISION NOT NULL CHECK (volume_mm3 > 0.0),
    condition           VARCHAR(64) NOT NULL,
    source_file         VARCHAR(128) NOT NULL,

    CONSTRAINT fk_station FOREIGN KEY (station_id) REFERENCES stations(station_id) ON DELETE CASCADE
);

-- Convert to a TimescaleDB hypertable partitioned by observed_at (7-day chunks)
-- if TimescaleDB extension is active on this database.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) THEN
        PERFORM create_hypertable(
            'raindrop_observations',
            'observed_at',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'raindrop_observations converted to TimescaleDB hypertable (7-day chunk interval).';
    ELSE
        RAISE NOTICE 'TimescaleDB extension not active; raindrop_observations created as standard PostgreSQL table.';
    END IF;
END
$$;

-- Create composite index for time-window and physical feature queries
CREATE INDEX IF NOT EXISTS idx_raindrop_station_time 
    ON raindrop_observations (station_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_raindrop_diameter_velocity 
    ON raindrop_observations (diameter_mm, velocity_m_s);

-- ----------------------------------------------------------------------------
-- 3. Rainfall Intensity Time-Series (Aggregated Disdrometer Records)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rainfall_intensity_records (
    period_start        TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    period_end          TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    station_id          VARCHAR(64) NOT NULL,
    rain_rate_mm_h      DOUBLE PRECISION NOT NULL CHECK (rain_rate_mm_h >= 0.0),
    total_drops         INTEGER NOT NULL CHECK (total_drops >= 0),
    mean_diameter_mm    DOUBLE PRECISION,
    mean_velocity_m_s   DOUBLE PRECISION,
    total_volume_mm3    DOUBLE PRECISION,
    condition           VARCHAR(64),

    CONSTRAINT pk_rainfall_intensity PRIMARY KEY (period_start, station_id),
    CONSTRAINT fk_intensity_station FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- ----------------------------------------------------------------------------
-- 4. REST API Prediction Audit Logs (FastAPI Backend Inference History)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_audit_logs (
    request_id          BIGSERIAL PRIMARY KEY,
    requested_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    input_diameter_mm   DOUBLE PRECISION NOT NULL CHECK (input_diameter_mm > 0.0),
    input_velocity_m_s  DOUBLE PRECISION NOT NULL CHECK (input_velocity_m_s > 0.0),
    formula_volume_mm3  DOUBLE PRECISION NOT NULL,
    xgb_volume_mm3      DOUBLE PRECISION,
    nn_volume_mm3       DOUBLE PRECISION,
    cluster_assigned    INTEGER,
    latency_ms          DOUBLE PRECISION,
    client_ip           VARCHAR(45)
);

CREATE INDEX IF NOT EXISTS idx_audit_time ON prediction_audit_logs (requested_at DESC);

-- ----------------------------------------------------------------------------
-- 5. Analytical Views for Reporting & Dashboards
-- ----------------------------------------------------------------------------

-- View: Hourly Drop Size Distribution (DSD) Summary
CREATE OR REPLACE VIEW v_hourly_dsd_summary AS
SELECT
    DATE_TRUNC('hour', observed_at) AS hour_window,
    station_id,
    condition,
    COUNT(*) AS drop_count,
    ROUND(AVG(diameter_mm)::numeric, 4) AS avg_diameter_mm,
    ROUND(AVG(velocity_m_s)::numeric, 4) AS avg_velocity_m_s,
    ROUND(SUM(volume_mm3)::numeric, 4) AS total_volume_mm3,
    COUNT(*) FILTER (WHERE diameter_mm > 2.6) AS extreme_drops_cluster2_count
FROM raindrop_observations
GROUP BY hour_window, station_id, condition;

-- View: Extreme Tail Raindrops (> 2.6 mm)
CREATE OR REPLACE VIEW v_extreme_raindrops AS
SELECT
    observed_at,
    station_id,
    condition,
    diameter_mm,
    velocity_m_s,
    volume_mm3
FROM raindrop_observations
WHERE diameter_mm > 2.6
ORDER BY diameter_mm DESC;
