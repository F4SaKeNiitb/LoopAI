# Lab test data with mappings for common tests
LAB_TEST_MAPPINGS = {
    "Hemoglobin": {
        "synonyms": [
            "hemoglobin", "hgb", "hb", "haemoglobin", "haemoglobin a", "hgb a", "hb a"
        ],
        "unit": "g/dL",
        "reference_range": {"min": 12.0, "max": 16.0},  # General for adults
        "critical_range": {"min": 5.0, "max": 20.0},
        "description": "Measures the oxygen-carrying protein in red blood cells"
    },
    "WBC": {
        "synonyms": [
            "white blood cell count", "wbc", "white cell count", "leukocyte count", "leucocyte count"
        ],
        "unit": "x10^9/L",
        "reference_range": {"min": 4.0, "max": 11.0},
        "critical_range": {"min": 1.0, "max": 30.0},
        "description": "Measures white blood cells, important for immune function"
    },
    "Creatinine": {
        "synonyms": [
            "creatinine", "creat", "cr", "creatinine serum", "serum creatinine"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 0.6, "max": 1.2},
        "critical_range": {"min": 0.3, "max": 5.0},
        "description": "Measures kidney function"
    },
    "TSH": {
        "synonyms": [
            "thyroid stimulating hormone", "tsh", "thyrotropin", "thyrotrophin"
        ],
        "unit": "mIU/L",
        "reference_range": {"min": 0.4, "max": 4.0},
        "critical_range": {"min": 0.01, "max": 100.0},
        "description": "Measures thyroid function"
    },
    "LDL": {
        "synonyms": [
            "ldl cholesterol", "ldl", "bad cholesterol", "ldl-c", "low density lipoprotein"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 0, "max": 100},  # Optimal is <100
        "critical_range": {"min": 0, "max": 190},
        "description": "Measures 'bad' cholesterol levels"
    },
    "HDL": {
        "synonyms": [
            "hdl cholesterol", "hdl", "good cholesterol", "hdl-c", "high density lipoprotein"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 40, "max": 100},  # Min 40 for men, 50 for women
        "critical_range": {"min": 20, "max": 100},
        "description": "Measures 'good' cholesterol levels"
    },
    "Total Cholesterol": {
        "synonyms": [
            "total cholesterol", "chol", "cholesterol", "total chol"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 125, "max": 200},
        "critical_range": {"min": 100, "max": 300},
        "description": "Measures total cholesterol levels"
    },
    "Triglycerides": {
        "synonyms": [
            "triglycerides", "trig", "tg", "tri", "trigs"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 0, "max": 150},
        "critical_range": {"min": 0, "max": 500},
        "description": "Measures fat in the blood"
    },
    "Glucose": {
        "synonyms": [
            "glucose", "blood glucose", "blood sugar", "glu", "bs", "blood sugar level"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 70, "max": 99},  # Fasting
        "critical_range": {"min": 40, "max": 400},
        "description": "Measures blood sugar levels"
    },
    "A1C": {
        "synonyms": [
            "hemoglobin a1c", "hba1c", "a1c", "glycated hemoglobin", "glycohemoglobin"
        ],
        "unit": "%",
        "reference_range": {"min": 4.0, "max": 5.6},
        "critical_range": {"min": 2.0, "max": 15.0},
        "description": "Measures average blood sugar over 2-3 months"
    },
    "Sodium": {
        "synonyms": [
            "sodium", "na", "na+", "serum sodium"
        ],
        "unit": "mEq/L",
        "reference_range": {"min": 136, "max": 145},
        "critical_range": {"min": 120, "max": 160},
        "description": "Electrolyte that helps maintain fluid balance"
    },
    "Potassium": {
        "synonyms": [
            "potassium", "k", "k+", "serum potassium"
        ],
        "unit": "mEq/L",
        "reference_range": {"min": 3.5, "max": 5.0},
        "critical_range": {"min": 2.5, "max": 6.5},
        "description": "Electrolyte important for heart and muscle function"
    },
    "Chloride": {
        "synonyms": [
            "chloride", "cl", "cl-", "serum chloride"
        ],
        "unit": "mEq/L",
        "reference_range": {"min": 98, "max": 107},
        "critical_range": {"min": 85, "max": 120},
        "description": "Electrolyte that helps maintain fluid balance"
    },
    "CO2": {
        "synonyms": [
            "carbon dioxide", "co2", "bicarbonate", "hco3", "total co2"
        ],
        "unit": "mEq/L",
        "reference_range": {"min": 23, "max": 29},
        "critical_range": {"min": 15, "max": 40},
        "description": "Measures acid-base balance"
    },
    "BUN": {
        "synonyms": [
            "blood urea nitrogen", "bun", "urea", "urea nitrogen"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 7, "max": 20},
        "critical_range": {"min": 5, "max": 50},
        "description": "Measures kidney function"
    },
    "ALT": {
        "synonyms": [
            "alt", "alanine aminotransferase", "sgpt", "alanine transaminase"
        ],
        "unit": "U/L",
        "reference_range": {"min": 7, "max": 56},
        "critical_range": {"min": 5, "max": 200},
        "description": "Liver enzyme test"
    },
    "AST": {
        "synonyms": [
            "ast", "aspartate aminotransferase", "sgot", "aspartate transaminase"
        ],
        "unit": "U/L",
        "reference_range": {"min": 10, "max": 40},
        "critical_range": {"min": 5, "max": 200},
        "description": "Liver enzyme test"
    },
    "Calcium": {
        "synonyms": [
            "calcium", "ca", "ca++", "serum calcium"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 8.5, "max": 10.5},
        "critical_range": {"min": 7.0, "max": 12.0},
        "description": "Important for bones and muscle function"
    },
    "Phosphorus": {
        "synonyms": [
            "phosphorus", "phos", "po4", "phosphate", "inorganic phosphorus"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 2.5, "max": 4.5},
        "critical_range": {"min": 1.0, "max": 8.0},
        "description": "Important for bones and energy production"
    },
    "Magnesium": {
        "synonyms": [
            "magnesium", "mg", "mg++", "serum magnesium"
        ],
        "unit": "mg/dL",
        "reference_range": {"min": 1.7, "max": 2.2},
        "critical_range": {"min": 1.0, "max": 4.0},
        "description": "Important for muscle and nerve function"
    }
}