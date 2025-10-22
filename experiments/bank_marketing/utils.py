from pathlib import Path

COLUMNS = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","income"
]

CONTINUOUS = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
CATEGORICAL = [c for c in COLUMNS if c not in CONTINUOUS and c != "income"]
TARGET = "income"

UCI_BASE = "https://archive.ics.uci.edu/static/public/2/adult.zip"

def project_paths(root: Path):
    root = root.resolve()
    data_cache = root / "data_cache"
    figures = root / "figures"
    return {"data_cache": data_cache, "figures": figures}
