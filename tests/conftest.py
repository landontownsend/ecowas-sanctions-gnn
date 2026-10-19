"""Shared test setup: settings, and a fresh in-memory warehouse built from the saved data."""

import os
from pathlib import Path

import duckdb
import pytest

from pipeline.config import load_config
from pipeline.ingest import ingest
from pipeline.transform import transform

# All paths in config.yaml are relative to the repository root
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def con(cfg):
    """An in-memory DuckDB warehouse with raw, staging, and feature tables (local mode)."""
    connection = duckdb.connect(":memory:")
    ingest(connection, cfg, live=False)
    transform(connection, cfg)
    yield connection
    connection.close()
