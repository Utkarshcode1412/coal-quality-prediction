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

## Project Structure
coal-quality-prediction/
│
├── app.py
├── train_model.py
├── coal_quality_data.csv
├── coal_quality_model.pkl
├── model_metrics.json
├── requirements.txt
├── README.md
└── .gitignore

## Tech Stack

Frontend & Dashboard  : Streamlit
Machine Learning      : Scikit-learn
Data Processing       : Pandas, NumPy
Visualization         : Plotly, Matplotlib
Model Serialization   : Joblib
Version Control       : Git & GitHub

## How to Run

## Installation & Execution

### 1. Clone the Repository
```bash 
git clone https://github.com/Utkarshcode1412/coal-quality-prediction.git
```
### 2. Navigate to Project Directory
``` bash
cd coal-quality-prediction
```

### 3. Create virtual environment

```bash
python -m venv venv
```

### 4. Activate virtual environment

For Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the model

```bash
python train_model.py
```

This will create:

```text
coal_quality_model.pkl
model_metrics.json
```

### 7. Run the dashboard

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

Evaluation metrics used:  
• R² Score
• Mean Absolute Error (MAE)
• Root Mean Squared Error (RMSE)
due to its superior prediction accuracy and robustness for coal quality estimation.

## Features Available in the Dashboard
• Single Sample Coal Quality Prediction
• Coal Grade Classification
• Quality Category Identification
• Blending & Dispatch Recommendations
• CSV Batch Prediction
• Interactive Data Analytics Dashboard
• Feature Importance Visualization
• Real-Time Sensor Simulation
• Downloadable Prediction Reports

## Hackathon Pitch

The system reduces dependency on delayed lab reports by predicting coal quality instantly from historical lab data, mine location, geological information, and optional sensor-like inputs. It supports faster decisions for blending, pricing, and dispatch.

## Live Demo

🔗 Deployed Application: https://coal-quality-prediction-dck4wrpzeegnedcisnadzu.streamlit.app/
