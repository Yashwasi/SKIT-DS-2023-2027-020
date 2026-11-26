"""
database/config.py
------------------
Database configuration module for Rain Drop Volume Prediction Using AI.
Loads database parameters from environment variables with safe defaults.
Masks credentials in logs and audit outputs to prevent credential leaks.
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    _project_root = Path(__file__).resolve().parent.parent
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
    else:
        load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class DatabaseConfig:
    """Immutable database configuration loaded from environment variables."""

    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "raindrop_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    sslmode: str = os.getenv("DB_SSLMODE", "prefer")

    pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))

    def get_sqlalchemy_url(self, driver: str = "psycopg2") -> str:
        """
        Build SQLAlchemy database connection URL.
        Example: postgresql+psycopg2://postgres:secret@localhost:5432/raindrop_db
        """
        driver_prefix = f"postgresql+{driver}" if driver else "postgresql"
        auth = self.user
        if self.password:
            pwd_quoted = urllib.parse.quote_plus(self.password)
            auth = f"{self.user}:{pwd_quoted}"
        return f"{driver_prefix}://{auth}@{self.host}:{self.port}/{self.name}"

    def get_safe_url(self) -> str:
        """
        Return the connection URL with the password masked for safe logging.
        Example: postgresql://postgres:***@localhost:5432/raindrop_db
        """
        masked_pwd = "***" if self.password else ""
        auth = f"{self.user}:{masked_pwd}" if masked_pwd else self.user
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.name}"

    def to_dict(self, mask_password: bool = True) -> Dict[str, Any]:
        """Convert config to dictionary, optionally masking the password."""
        return {
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "user": self.user,
            "password": "***" if (mask_password and self.password) else self.password,
            "sslmode": self.sslmode,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
        }


# Singleton default configuration instance
db_config = DatabaseConfig()
