# Patient Survival Predictor 🏥

A machine learning script that predicts in-hospital patient mortality (`hospital_death`) from ICU admission data, using scikit-learn's Logistic Regression.

## Overview

`patient_survival_predictor.py` loads a patient dataset (`dataset.csv`), cleans and preprocesses it, then trains a logistic regression classifier to predict whether a patient survives their hospital stay.

### Preprocessing steps

1. Drops a stray `Unnamed: 83` column left over from the source CSV export.
2. Factorizes (label-encodes) any categorical/object columns into numeric codes.
3. Imputes missing values in every column using the column mean (`SimpleImputer(strategy="mean")`).
4. Splits the data 80/20 into train/test sets (`random_state=42` for reproducibility).
5. Scales features with `StandardScaler`.

### Model

- **Algorithm:** Logistic Regression (`C=0.1`, `solver="saga"`, `penalty="l2"`)
- **Target:** `hospital_death` (binary — survived vs. did not survive)
- **Accuracy:** **92.2%** on the held-out test set

## Getting Started

### Prerequisites

- Python 3.8+
- pandas, scikit-learn

```bash
pip install pandas scikit-learn
```

### Data

This repo doesn't include the dataset. Place a `dataset.csv` file in the project root before running the script. It should contain a `hospital_death` target column plus the ICU/clinical feature columns the model expects (the column structure matches datasets like the WiDS Datathon patient-survival / MIT GOSSIS ICU dataset).

### Run

```bash
python patient_survival_predictor.py
```

This prints the model's accuracy on the test set.

## Project Structure

```
patient-surival-predictor/
└── patient_survival_predictor.py   # Data loading, preprocessing, training, and evaluation
```

## Disclaimer

This is an educational/portfolio project. It is **not validated for clinical use** and should not be used to inform real medical or treatment decisions.

## License

No license specified yet — add one (e.g. MIT) if you plan to share or accept contributions to this project.
