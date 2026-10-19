"""Load config.yaml, the single settings file for the project."""

import yaml


def load_config(path="config.yaml"):
    """Read config.yaml and return it as a Python dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)
