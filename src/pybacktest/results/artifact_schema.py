"""Canonical ordered file graph for persisted result artifacts."""

ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "summary.json",
    "equity.parquet",
    "positions.parquet",
    "orders.parquet",
    "fills.parquet",
    "events.parquet",
)
ARTIFACT_TRUST_FILES = ("manifest.json", "manifest.sha256")
ARTIFACT_ALL_FILES = (*ARTIFACT_PAYLOAD_FILES, *ARTIFACT_TRUST_FILES)

__all__ = [
    "ARTIFACT_ALL_FILES",
    "ARTIFACT_PAYLOAD_FILES",
    "ARTIFACT_TRUST_FILES",
]
