import json
import random
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Mining AI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

page_bg = """
<style>

/* Main App Background */
.stApp {
    background-image: linear-gradient(
        rgba(0,0,0,0.75),
        rgba(0,0,0,0.75)
    ),
    url("https://images.unsplash.com/photo-1593250718797-ce0f8c0728a5?w=900&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.92);
    backdrop-filter: blur(10px);
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main Content */
.main {
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #FFB000 !important;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 18px;
    border-radius: 16px;
    backdrop-filter: blur(8px);
}

/* Custom Cards */
.custom-card {
    background: rgba(30, 41, 59, 0.75);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #FFB000;
    color: black;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    padding: 10px 20px;
}

/* Radio Buttons */
.stRadio label {
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: rgba(17,24,39,0.85);
    border-radius: 12px;
}

/* Input Fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div {
    background-color: rgba(255,255,255,0.08);
    color: white !important;
}

/* Hide Streamlit Branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = Path("coal_quality_data.csv")
MODEL_PATH = Path("coal_quality_model.pkl")
METRICS_PATH = Path("model_metrics.json")

FEATURE_COLUMNS = [
    "mine_location",
    "moisture",
    "ash",
    "volatile_matter",
    "fixed_carbon",
    "sulphur",
    "temperature",
    "depth",
]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Single Prediction",
        "📂 CSV Batch Prediction",
        "📊 Analytics Dashboard",
        "📡 Sensor Simulation",
        "ℹ️ About Project",
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
### 🚀 Project Highlights

- AI-Powered Prediction
- Coal Grade Classification
- Decision Support System
- Batch CSV Analysis
- Explainable AI
- IoT-Ready Architecture
"""
)

# =========================================================
# LOAD MODEL & DATA
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. First run: python train_model.py")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Dataset file not found.")
        st.stop()
    return pd.read_csv(DATA_PATH)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_grade(gcv: float) -> str:

    if gcv >= 7000:
        return "G1"
    elif gcv >= 6700:
        return "G2"
    elif gcv >= 6400:
        return "G3"
    elif gcv >= 6100:
        return "G4"
    elif gcv >= 5800:
        return "G5"
    elif gcv >= 5500:
        return "G6"
    elif gcv >= 5200:
        return "G7"
    elif gcv >= 4900:
        return "G8"
    elif gcv >= 4600:
        return "G9"
    elif gcv >= 4300:
        return "G10"
    elif gcv >= 4000:
        return "G11"
    elif gcv >= 3700:
        return "G12"
    elif gcv >= 3400:
        return "G13"

    return "G14"


def get_quality_category(gcv: float) -> str:

    if gcv >= 7000:
        return "High Quality"

    elif gcv >= 5800:
        return "Medium Quality"

    return "Low Quality"


def get_price_category(gcv: float) -> str:

    if gcv >= 7000:
        return "Premium Price Category"

    elif gcv >= 5800:
        return "Standard Price Category"

    return "Discount / Blending Required"


def get_recommendation(gcv, ash, moisture, sulphur):

    suggestions = []

    if gcv >= 7000:
        suggestions.append(
            "High-grade coal suitable for premium industrial usage."
        )

    elif gcv >= 5800:
        suggestions.append(
            "Medium-grade coal suitable for thermal power plants."
        )

    else:
        suggestions.append(
            "Low-grade coal. Blending recommended before dispatch."
        )

    if ash > 20:
        suggestions.append(
            "High ash content detected. Blend with low-ash coal."
        )

    if moisture > 10:
        suggestions.append(
            "High moisture detected. Drying/blending recommended."
        )

    if sulphur > 3:
        suggestions.append(
            "High sulphur level detected. Environmental caution advised."
        )

    return " ".join(suggestions)


def get_blending_recommendation(gcv, ash, moisture, sulphur):

    current_grade = get_grade(gcv)

    if gcv < 3400:
        blend_with = "G4 coal"
        target_grade = "G8"
        ratio = "60:40"
        target_gcv = 5000

    elif gcv < 4300:
        blend_with = "G5 coal"
        target_grade = "G7"
        ratio = "65:35"
        target_gcv = 5200

    elif gcv < 5200:
        blend_with = "G3 coal"
        target_grade = "G5"
        ratio = "70:30"
        target_gcv = 5800

    elif gcv < 5800:
        blend_with = "G2 coal"
        target_grade = "G4"
        ratio = "75:25"
        target_gcv = 6200

    else:
        return f"Current grade is {current_grade}. No blending required."

    return f"""
Current Grade: {current_grade}

Blend With: {blend_with}

Ratio: {ratio}

Expected Grade: {target_grade}

Expected GCV: {target_gcv} kcal/kg
"""


def predict_single(model, input_data):

    input_df = pd.DataFrame([input_data])

    predicted_gcv = float(model.predict(input_df)[0])

    return round(predicted_gcv, 2)

# =========================================================
# HEADER
# =========================================================

def show_header():

    st.title(
        "⛏️ AI-Based Coal Sample Analysis & Quality Prediction System"
    )

    st.caption(
        "Smart Mining • Predictive Analytics • Decision Support Platform"
    )

# =========================================================
# HOME PAGE
# =========================================================

def home_page(df, metrics):

    st.subheader("Project Objective")

    st.write(
        """
This system predicts coal quality instantly using Machine Learning
and converts predictions into actionable operational recommendations.
"""
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Rows", len(df))
    c2.metric("Mine Locations", df["mine_location"].nunique())
    c3.metric("Best Model", metrics.get("final_model", "Random Forest"))
    c4.metric("R² Score", metrics.get("r2_score", "N/A"))

    st.subheader("System Workflow")

    st.info(
        """
Coal Sample Data → ML Model → GCV Prediction →
Coal Grade Classification → Decision Support
"""
    )

    st.subheader("Sample Dataset")

    st.dataframe(df.head(10), use_container_width=True)

# =========================================================
# SINGLE PREDICTION
# =========================================================

def single_prediction_page(df, model):

    st.subheader("🔮 Single Coal Sample Prediction")

    mine_locations = sorted(df["mine_location"].unique().tolist())

    col1, col2 = st.columns(2)

    with col1:

        mine_location = st.selectbox(
            "Mine Location",
            mine_locations
        )

        moisture = st.number_input(
            "Moisture (%)",
            0.0,
            30.0,
            7.0
        )

        ash = st.number_input(
            "Ash (%)",
            0.0,
            50.0,
            15.0
        )

        volatile_matter = st.number_input(
            "Volatile Matter (%)",
            0.0,
            60.0,
            30.0
        )

    with col2:

        fixed_carbon = st.number_input(
            "Fixed Carbon (%)",
            0.0,
            90.0,
            48.0
        )

        sulphur = st.number_input(
            "Sulphur (%)",
            0.0,
            10.0,
            2.0
        )

        temperature = st.number_input(
            "Process Temperature",
            0.0,
            2000.0,
            1250.0
        )

        depth = st.number_input(
            "Mining Depth",
            0.0,
            1000.0,
            150.0
        )

    input_data = {
        "mine_location": mine_location,
        "moisture": moisture,
        "ash": ash,
        "volatile_matter": volatile_matter,
        "fixed_carbon": fixed_carbon,
        "sulphur": sulphur,
        "temperature": temperature,
        "depth": depth,
    }

    if st.button("Predict Coal Quality", type="primary"):

        predicted_gcv = predict_single(model, input_data)

        grade = get_grade(predicted_gcv)

        quality = get_quality_category(predicted_gcv)

        price = get_price_category(predicted_gcv)

        recommendation = get_recommendation(
            predicted_gcv,
            ash,
            moisture,
            sulphur
        )

        blending = get_blending_recommendation(
            predicted_gcv,
            ash,
            moisture,
            sulphur
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Predicted GCV",
            f"{predicted_gcv} kcal/kg"
        )

        c2.metric(
            "Coal Grade",
            grade
        )

        c3.metric(
            "Quality",
            quality
        )

        c4.metric(
            "Pricing",
            price
        )

        st.success(recommendation)

        st.subheader("Blending Recommendation")

        st.info(blending)

def batch_prediction_page(model):
    st.subheader("CSV Batch Prediction")
    st.write("Upload a CSV file with the same input columns used for training.")

    uploaded_file = st.file_uploader("Upload Coal Sample CSV", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        try:
            result_df = add_prediction_columns(uploaded_df, model)
            st.success("Batch prediction completed.")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Report",
                data=csv,
                file_name="coal_quality_prediction_report.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(str(e))

def add_prediction_columns(df: pd.DataFrame, model):

    prediction_df = df.copy()

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in prediction_df.columns
    ]

    if missing:
        raise ValueError(
            f"Uploaded CSV is missing columns: {missing}"
        )

    prediction_df["predicted_gcv"] = model.predict(
        prediction_df[FEATURE_COLUMNS]
    ).round(2)

    prediction_df["grade"] = prediction_df[
        "predicted_gcv"
    ].apply(get_grade)

    prediction_df["quality_category"] = prediction_df[
        "predicted_gcv"
    ].apply(get_quality_category)

    prediction_df["price_category"] = prediction_df[
        "predicted_gcv"
    ].apply(get_price_category)

    prediction_df["recommendation"] = prediction_df.apply(
        lambda row: get_recommendation(
            row["predicted_gcv"],
            row["ash"],
            row["moisture"],
            row["sulphur"]
        ),
        axis=1
    )

    return prediction_df

# =========================================================
# ANALYTICS PAGE
# =========================================================

def analytics_page(df, model):
    st.subheader("Coal Quality Analytics")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.scatter(
            df,
            x="ash",
            y="gcv",
            color="mine_location",
            title="Ash % vs GCV",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            df,
            x="moisture",
            y="gcv",
            color="mine_location",
            title="Moisture % vs GCV",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Mine-wise Average GCV")
    mine_quality = df.groupby("mine_location", as_index=False)["gcv"].mean().sort_values("gcv", ascending=False)
    fig3 = px.bar(mine_quality, x="mine_location", y="gcv", title="Average GCV by Mine Location")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Importance")
    try:
        regressor = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        fig4 = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Model Feature Importance")
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)
    except Exception:
        st.warning("Feature importance is available only for tree-based models.")

# =========================================================
# SENSOR SIMULATION
# =========================================================

def sensor_simulation_page(df, model):

    st.subheader("📡 Sensor Fusion Simulation")

    if st.button("Generate Live Sensor Sample", type="primary"):

        sample = {
            "mine_location": random.choice(
                df["mine_location"].unique().tolist()
            ),
            "moisture": round(random.uniform(3, 12), 2),
            "ash": round(random.uniform(5, 25), 2),
            "volatile_matter": round(random.uniform(18, 42), 2),
            "sulphur": round(random.uniform(0.4, 3.8), 2),
            "temperature": round(random.uniform(1100, 1480), 1),
            "depth": round(random.uniform(20, 310), 1),
        }

        sample["fixed_carbon"] = round(
            100
            - sample["moisture"]
            - sample["ash"]
            - sample["volatile_matter"],
            2
        )

        predicted_gcv = predict_single(model, sample)

        grade = get_grade(predicted_gcv)

        st.json(sample)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Live Predicted GCV",
            f"{predicted_gcv} kcal/kg"
        )

        c2.metric(
            "Predicted Grade",
            grade
        )

        c3.metric(
            "Quality",
            get_quality_category(predicted_gcv)
        )

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

def about_project_page():

    st.subheader("ℹ️ About Project")

    st.markdown(
        """
<div class="custom-card">

<h3>⛏️ AI-Powered Coal Quality Prediction Platform</h3>

<p>
Coal quality assessment in mining industries is traditionally dependent on laboratory analysis,
which is time-consuming, resource-intensive, and delays operational decision-making.

This project proposes an AI-powered Coal Quality Prediction and Decision Support System
capable of predicting coal quality instantly using machine learning techniques.

The system utilizes historical coal sample data along with operational and geological
parameters to predict Gross Calorific Value (GCV) and classify coal grades automatically.

A Random Forest Regression model is used to provide accurate predictions, while an interactive
Streamlit dashboard enables real-time analysis, visualization, and recommendation generation.

The system also includes an IoT-ready sensor simulation module to demonstrate future integration
with live mining sensors and conveyor monitoring systems.

The proposed solution aims to reduce dependency on delayed lab testing,
improve operational efficiency,
support intelligent decision-making,
and contribute toward digital transformation in smart mining ecosystems.
</p>

</div>
""",
        unsafe_allow_html=True
    )

    st.subheader("🎯 Key Features")

    features = [
        "AI-Based GCV Prediction",
        "Coal Grade Classification",
        "Blending Recommendation",
        "Pricing Recommendation",
        "IoT-Ready Architecture",
        "Batch CSV Prediction",
        "Interactive Analytics Dashboard",
    ]

    for feature in features:
        st.markdown(f"✅ {feature}")

    st.subheader("🚀 Innovation Highlights")

    innovations = [
        "Real-time AI prediction system",
        "Decision support recommendations",
        "Explainable AI architecture",
        "Industrial mining use-case",
        "Sensor fusion simulation",
    ]

    for item in innovations:
        st.markdown(f"🔹 {item}")

    st.subheader("🏗️ System Architecture")

    st.info(
        """
Coal Sample Data
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
GCV Prediction
        ↓
Coal Grade Classification
        ↓
Decision Support System
        ↓
Dashboard Visualization
"""
    )

    st.subheader("🏭 Industry Impact")

    st.success(
        """
• Reduces dependency on delayed laboratory testing

• Improves operational efficiency

• Supports faster dispatch decisions

• Helps optimize coal blending strategies

• Enables AI-driven mining operations
"""
    )

    st.subheader("👨‍💻 Project Team")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
<div class="custom-card">

<h3>Utkarsh Pawar</h3>

<p><b>Role:</b> AI Model Development & System Architecture</p>

<p><b>Department:</b> Electrical Engineering</p>

<p>
Developed Machine Learning pipeline,
prediction system,
dashboard integration,
and deployment workflow.
</p>

</div>
""",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
<div class="custom-card">

<h3>Aniket Sonawane</h3>

<p><b>Role:</b> Frontend Development & Testing</p>

<p><b>Department:</b> Mining Engineering</p>

<p>
Worked on frontend enhancement,
dashboard testing,
documentation,
and presentation support.
</p>

</div>
""",
            unsafe_allow_html=True
        )

    st.subheader("🔗 Project Links")

    st.markdown(
        """
- [GitHub Repository](https://github.com/Utkarshcode1412/coal-quality-prediction/)
""",
        unsafe_allow_html=True
    )
# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    show_header()

    df = load_data()

    model = load_model()

    if METRICS_PATH.exists():

        with open(METRICS_PATH, "r") as f:

            metrics = json.load(f)

    else:

        metrics = {}

    if page == "🏠 Home":

        home_page(df, metrics)

    elif page == "🔮 Single Prediction":

        single_prediction_page(df, model)
        
    elif page == "📂 CSV Batch Prediction":

        batch_prediction_page(model)

    elif page == "📊 Analytics Dashboard":

        analytics_page(df, model)

    elif page == "📡 Sensor Simulation":

        sensor_simulation_page(df, model)

    elif page == "ℹ️ About Project":

        about_project_page()

# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    main()
