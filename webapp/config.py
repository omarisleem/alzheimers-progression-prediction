"""Feature definitions aligned with the modeling notebook."""

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

# UI metadata: label, help, input kind
FEATURE_UI = {
    "PTGENDER": {
        "label": "Sex",
        "help": "1 = Male, 2 = Female (ADNI coding)",
        "kind": "select",
        "options": {1: "Male", 2: "Female"},
    },
    "PTEDUCAT": {
        "label": "Years of education",
        "help": "Years of formal education",
        "kind": "number",
    },
    "PTMARRY": {
        "label": "Marital status",
        "help": "ADNI marital status code",
        "kind": "select",
        "options": {
            1: "Married",
            2: "Widowed",
            3: "Divorced",
            4: "Separated",
            5: "Never married",
            6: "Unknown / other",
        },
    },
    "APOE4_COUNT": {
        "label": "APOE ε4 allele count",
        "help": "0, 1, or 2 copies",
        "kind": "select",
        "options": {0: "0", 1: "1", 2: "2"},
    },
    "AGE": {
        "label": "Age (years)",
        "help": "Age at visit",
        "kind": "number",
    },
    "LIMMTOTAL": {
        "label": "Logical memory (immediate)",
        "help": "Wechsler Logical Memory — immediate recall total",
        "kind": "number",
    },
    "LDELTOTAL": {
        "label": "Logical memory (delayed)",
        "help": "Wechsler Logical Memory — delayed recall total",
        "kind": "number",
    },
    "AVDEL30MIN": {
        "label": "RAVLT delayed recall (30 min)",
        "help": "Rey AVLT — 30-minute delayed recall",
        "kind": "number",
    },
    "AVDELTOT": {
        "label": "RAVLT recognition total",
        "help": "Rey AVLT recognition score",
        "kind": "number",
    },
    "CATANIMSC": {
        "label": "Category fluency (animals)",
        "help": "Number of animals named in one minute",
        "kind": "number",
    },
    "MMSCORE": {
        "label": "MMSE total score",
        "help": "Mini-Mental State Examination (0–30)",
        "kind": "number",
    },
    "CDRSB": {
        "label": "CDR sum of boxes",
        "help": "Clinical Dementia Rating — sum of boxes",
        "kind": "number",
    },
    "FAQTOTAL": {
        "label": "FAQ total score",
        "help": "Functional Activities Questionnaire total",
        "kind": "number",
    },
    "TOTSCORE": {
        "label": "ADAS-Cog total score",
        "help": "Alzheimer's Disease Assessment Scale — cognitive subscale",
        "kind": "number",
    },
    "TOTAL13": {
        "label": "ADAS-Cog total (13 items)",
        "help": "ADAS-Cog 13-item total",
        "kind": "number",
    },
    "BRAINVOL": {
        "label": "Brain volume",
        "help": "Whole-brain volume (imaging feature from cohort)",
        "kind": "number",
    },
    "VENTVOL": {
        "label": "Ventricular volume",
        "help": "Ventricular volume (imaging feature from cohort)",
        "kind": "number",
    },
    "TOTAL_HIPPO": {
        "label": "Total hippocampal volume",
        "help": "Combined hippocampal volume",
        "kind": "number",
    },
    "RAVLT_AVERAGE": {
        "label": "RAVLT learning average",
        "help": "Average across RAVLT learning trials",
        "kind": "number",
    },
}
