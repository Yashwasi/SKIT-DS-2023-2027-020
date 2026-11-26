"""
database/models.py
------------------
SQLAlchemy ORM models and dataclass equivalents corresponding to database/schema.sql.
Includes:
- Station (environmental disdrometer monitoring stations)
- RaindropObservation (individual disdrometer particle records - Timescale hypertable)
- RainfallIntensityRecord (aggregated rain rate / intensity intervals)
- PredictionAuditLog (FastAPI backend volume prediction inference audit history)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from sqlalchemy import (
        BigInteger,
        CheckConstraint,
        Column,
        DateTime,
        Float,
        ForeignKey,
        Index,
        Integer,
        String,
        func,
    )
    from sqlalchemy.orm import declarative_base, relationship
    _HAS_SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    _HAS_SQLALCHEMY = False
    Base = object  # type: ignore[misc,assignment]


if _HAS_SQLALCHEMY:

    class Station(Base):
        """Disdrometer deployment stations and tree canopy environmental characteristics."""

        __tablename__ = "stations"

        station_id = Column(String(64), primary_key=True)
        station_name = Column(String(128), nullable=False)
        canopy_type = Column(String(64), nullable=False)
        instrument_type = Column(String(64), default="Optical Disdrometer")
        latitude = Column(Float, nullable=True)
        longitude = Column(Float, nullable=True)
        elevation_m = Column(Float, nullable=True)
        created_at = Column(DateTime(timezone=True), server_default=func.now())

        # Relationships
        observations = relationship("RaindropObservation", back_populates="station", cascade="all, delete-orphan")
        intensity_records = relationship("RainfallIntensityRecord", back_populates="station", cascade="all, delete-orphan")

        def __repr__(self) -> str:
            return f"<Station(id={self.station_id!r}, name={self.station_name!r}, canopy={self.canopy_type!r})>"

    class RaindropObservation(Base):
        """
        Individual raindrop observations recorded by optical disdrometer.
        Converts to TimescaleDB hypertable partitioned by observed_at in 7-day chunks.
        """

        __tablename__ = "raindrop_observations"
        __table_args__ = (
            CheckConstraint("diameter_mm > 0.0 AND diameter_mm <= 15.0", name="chk_diameter_range"),
            CheckConstraint("velocity_m_s > 0.0 AND velocity_m_s <= 25.0", name="chk_velocity_range"),
            CheckConstraint("volume_mm3 > 0.0", name="chk_volume_positive"),
            Index("idx_raindrop_station_time", "station_id", "observed_at"),
            Index("idx_raindrop_diameter_velocity", "diameter_mm", "velocity_m_s"),
        )

        # Primary key observation_id guarantees unique identification per drop.
        # In Timescale hypertables, observed_at is part of partition key.
        observation_id = Column(
            BigInteger().with_variant(Integer, "sqlite"),
            primary_key=True,
            autoincrement=True,
        )
        observed_at = Column(DateTime, nullable=False, index=True)
        station_id = Column(String(64), ForeignKey("stations.station_id", ondelete="CASCADE"), nullable=False)
        diameter_mm = Column(Float, nullable=False)
        velocity_m_s = Column(Float, nullable=False)
        volume_mm3 = Column(Float, nullable=False)
        condition = Column(String(64), nullable=False)
        source_file = Column(String(128), nullable=False)

        station = relationship("Station", back_populates="observations")

        def __repr__(self) -> str:
            return (
                f"<RaindropObservation(station={self.station_id!r}, time={self.observed_at}, "
                f"d={self.diameter_mm:.3f}mm, v={self.velocity_m_s:.2f}m/s, vol={self.volume_mm3:.4f}mm³)>"
            )

    class RainfallIntensityRecord(Base):
        """Aggregated disdrometer time-series records for rain rate analysis."""

        __tablename__ = "rainfall_intensity_records"
        __table_args__ = (
            CheckConstraint("rain_rate_mm_h >= 0.0", name="chk_rain_rate_positive"),
            CheckConstraint("total_drops >= 0", name="chk_drops_positive"),
        )

        period_start = Column(DateTime, primary_key=True)
        station_id = Column(String(64), ForeignKey("stations.station_id", ondelete="CASCADE"), primary_key=True)
        period_end = Column(DateTime, nullable=False)
        rain_rate_mm_h = Column(Float, nullable=False)
        total_drops = Column(Integer, nullable=False)
        mean_diameter_mm = Column(Float, nullable=True)
        mean_velocity_m_s = Column(Float, nullable=True)
        total_volume_mm3 = Column(Float, nullable=True)
        condition = Column(String(64), nullable=True)

        station = relationship("Station", back_populates="intensity_records")

        def __repr__(self) -> str:
            return f"<RainfallIntensityRecord(station={self.station_id!r}, start={self.period_start}, rate={self.rain_rate_mm_h:.2f}mm/h)>"

    class PredictionAuditLog(Base):
        """Audit trail of volume prediction requests received by FastAPI backend."""

        __tablename__ = "prediction_audit_logs"
        __table_args__ = (
            CheckConstraint("input_diameter_mm > 0.0", name="chk_audit_diam_positive"),
            CheckConstraint("input_velocity_m_s > 0.0", name="chk_audit_vel_positive"),
            Index("idx_audit_time", "requested_at"),
        )

        request_id = Column(
            BigInteger().with_variant(Integer, "sqlite"),
            primary_key=True,
            autoincrement=True,
        )
        requested_at = Column(DateTime(timezone=True), server_default=func.now())
        input_diameter_mm = Column(Float, nullable=False)
        input_velocity_m_s = Column(Float, nullable=False)
        formula_volume_mm3 = Column(Float, nullable=False)
        xgb_volume_mm3 = Column(Float, nullable=True)
        nn_volume_mm3 = Column(Float, nullable=True)
        cluster_assigned = Column(Integer, nullable=True)
        latency_ms = Column(Float, nullable=True)
        client_ip = Column(String(45), nullable=True)

        def __repr__(self) -> str:
            return f"<PredictionAuditLog(id={self.request_id}, diam={self.input_diameter_mm}, vol={self.formula_volume_mm3})>"


else:
    # Standalone dataclass fallbacks for environments without SQLAlchemy installed
    @dataclass
    class Station:  # type: ignore[no-redef]
        station_id: str
        station_name: str
        canopy_type: str
        instrument_type: str = "Optical Disdrometer"
        latitude: Optional[float] = None
        longitude: Optional[float] = None
        elevation_m: Optional[float] = None
        created_at: Optional[datetime.datetime] = None

    @dataclass
    class RaindropObservation:  # type: ignore[no-redef]
        observed_at: datetime.datetime
        station_id: str
        diameter_mm: float
        velocity_m_s: float
        volume_mm3: float
        condition: str
        source_file: str
        observation_id: Optional[int] = None

    @dataclass
    class RainfallIntensityRecord:  # type: ignore[no-redef]
        period_start: datetime.datetime
        station_id: str
        period_end: datetime.datetime
        rain_rate_mm_h: float
        total_drops: int
        mean_diameter_mm: Optional[float] = None
        mean_velocity_m_s: Optional[float] = None
        total_volume_mm3: Optional[float] = None
        condition: Optional[str] = None

    @dataclass
    class PredictionAuditLog:  # type: ignore[no-redef]
        input_diameter_mm: float
        input_velocity_m_s: float
        formula_volume_mm3: float
        request_id: Optional[int] = None
        requested_at: Optional[datetime.datetime] = None
        xgb_volume_mm3: Optional[float] = None
        nn_volume_mm3: Optional[float] = None
        cluster_assigned: Optional[int] = None
        latency_ms: Optional[float] = None
        client_ip: Optional[str] = None
