from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "raw_data.csv"
INTERIM_DATA_PATH = ROOT_DIR / "data" / "interim" / "fraud_cleaned.csv"
REPORTS_DIR = ROOT_DIR / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "data_validation.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda.json"
EDA_MD_PATH = REPORTS_DIR / "eda.md"
EDA_BY_TYPE_PATH = REPORTS_DIR / "fraud_by_type.csv"
EDA_BY_HOUR_PATH = REPORTS_DIR / "fraud_by_hour.csv"
RANDOM_STATE = 42
COST_FP = 1.0
COST_FN = 10.0
