"""
ECOWAS sanctions risk data pipeline.

Steps, in order:
  1. ingest     load raw data into DuckDB (World Bank API or saved snapshot,
                V-Dem release file, hand-coded ECOWAS events, reference data)
  2. transform  SQL models: raw -> staging -> features
  3. checks     data quality checks; the run stops if any check fails
  4. export     write the study-period features and the forecast-year inputs
                to data/processed/ for the model scripts

Run from the repository root:
  python -m pipeline.run          # uses the saved World Bank snapshot
  python -m pipeline.run --live   # calls the World Bank API
"""
