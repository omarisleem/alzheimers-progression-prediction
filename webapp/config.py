"""Feature columns aligned with df_clean_AD.csv / modeling notebook."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "Processed" / "df_clean_AD.csv"
TARGET_COLUMN = "PROGRESSOR"

MODEL_FEATURES = [
    "PTGENDER",
    "PTEDUCAT",
    "PTMARRY",
    "APOE4_COUNT",
    "AGE",
    "LIMMTOTAL",
    "LDELTOTAL",
    "AVDEL30MIN",
    "AVDELTOT",
    "CATANIMSC",
    "MMSCORE",
    "CDRSB",
    "FAQTOTAL",
    "TOTSCORE",
    "TOTAL13",
    "BRAINVOL",
    "VENTVOL",
    "TOTAL_HIPPO",
    "RAVLT_AVERAGE",
]

SCALE_FEATURES = [f for f in MODEL_FEATURES if f != "PTGENDER"]

# Discrete columns in the processed table (integer-coded)
INTEGER_FEATURES = {"PTGENDER", "PTMARRY", "APOE4_COUNT"}

TARGET_LABELS = {0: "Non-progressor", 1: "Progressor"}
