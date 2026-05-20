# AI-Based Coal Sample Analysis & Quality Prediction System

This project predicts coal quality using historical coal sample data and optional sensor-like inputs.

## Features

- GCV prediction using machine learning
- Coal grade classification
- Quality category prediction
- Pricing and dispatch recommendation
- CSV batch prediction
- Feature importance visualization
- Sensor fusion simulation dashboard

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## How to Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

For Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This will create:

```text
coal_quality_model.pkl
model_metrics.json
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

## Dataset Columns

```text
sample_id
mine_location
moisture
ash
volatile_matter
fixed_carbon
sulphur
temperature
depth
gcv
```

## Model

The training script compares:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best model is saved and used by the Streamlit app.

## Hackathon Pitch

The system reduces dependency on delayed lab reports by predicting coal quality instantly from historical lab data, mine location, geological information, and optional sensor-like inputs. It supports faster decisions for blending, pricing, and dispatch.
