import json
import random
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


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


st.set_page_config(
    page_title="AI Coal Quality Prediction",
    page_icon="⛏️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. First run: python train_model.py")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Dataset file not found. Keep coal_quality_data.csv in this folder.")
        st.stop()
    return pd.read_csv(DATA_PATH)


def get_grade(gcv: float) -> str:
    if gcv > 7000:
        return "G1"
    elif gcv > 6700:
        return "G2"
    elif gcv > 6400:
        return "G3"
    elif gcv > 6100:
        return "G4"
    elif gcv > 5800:
        return "G5"
    elif gcv > 5500:
        return "G6"
    elif gcv > 5200:
        return "G7"
    return "Low Grade"


def get_quality_category(gcv: float) -> str:
    if gcv >= 7000:
        return "High Quality"
    elif gcv >= 5800:
        return "Medium Quality"
    return "Low Quality"


def get_recommendation(gcv: float, ash: float, moisture: float, sulphur: float) -> str:
    suggestions = []

    if gcv >= 7000:
        suggestions.append("High-grade coal. Suitable for premium industrial usage and direct dispatch.")
    elif gcv >= 5800:
        suggestions.append("Medium-grade coal. Suitable for thermal power plant usage.")
    else:
        suggestions.append("Low-grade coal. Blending is recommended before dispatch.")

    if ash > 20:
        suggestions.append("Ash content is high. Blend with low-ash coal to improve quality.")
    if moisture > 10:
        suggestions.append("Moisture is high. Drying or blending can improve effective calorific value.")
    if sulphur > 3:
        suggestions.append("Sulphur is high. Use carefully because of environmental restrictions.")

    if not suggestions:
        suggestions.append("Coal quality is acceptable for normal dispatch.")

    return " ".join(suggestions)


def get_price_category(gcv: float) -> str:
    if gcv >= 7000:
        return "Premium Price Category"
    elif gcv >= 5800:
        return "Standard Price Category"
    return "Discount / Blending Required"


def predict_single(model, input_data: dict):
    input_df = pd.DataFrame([input_data])
    predicted_gcv = float(model.predict(input_df)[0])
    return round(predicted_gcv, 2)


def add_prediction_columns(df: pd.DataFrame, model):
    prediction_df = df.copy()

    missing = [col for col in FEATURE_COLUMNS if col not in prediction_df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing columns: {missing}")

    prediction_df["predicted_gcv"] = model.predict(prediction_df[FEATURE_COLUMNS]).round(2)
    prediction_df["grade"] = prediction_df["predicted_gcv"].apply(get_grade)
    prediction_df["quality_category"] = prediction_df["predicted_gcv"].apply(get_quality_category)
    prediction_df["price_category"] = prediction_df["predicted_gcv"].apply(get_price_category)
    prediction_df["recommendation"] = prediction_df.apply(
        lambda row: get_recommendation(
            row["predicted_gcv"], row["ash"], row["moisture"], row["sulphur"]
        ),
        axis=1,
    )
    return prediction_df


def show_header():
    st.title("⛏️ AI-Based Coal Sample Analysis & Quality Prediction System")
    st.caption(
        "Predict coal GCV, classify coal grade, and generate blending, pricing, and dispatch recommendations."
    )


def home_page(df, metrics):
    st.subheader("Project Objective")
    st.write(
        """
        This system predicts coal quality instantly using historical lab data, mine location,
        geological information, and optional sensor-like inputs. The model predicts GCV and
        converts it into coal grade, quality category, pricing class, and dispatch recommendation.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", len(df))
    c2.metric("Mine Locations", df["mine_location"].nunique())
    c3.metric("Best Model", metrics.get("final_model", "Random Forest"))
    c4.metric("R² Score", metrics.get("r2_score", "N/A"))

    st.subheader("System Workflow")
    st.info(
        "Coal Sample Data → Preprocessing → ML Model → GCV Prediction → Grade Classification → Decision Support"
    )

    st.subheader("Sample Dataset")
    st.dataframe(df.head(10), use_container_width=True)


def single_prediction_page(df, model):
    st.subheader("Single Coal Sample Prediction")

    mine_locations = sorted(df["mine_location"].unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        mine_location = st.selectbox("Mine Location", mine_locations)
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=30.0, value=7.0, step=0.1)
        ash = st.number_input("Ash (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        volatile_matter = st.number_input("Volatile Matter (%)", min_value=0.0, max_value=60.0, value=30.0, step=0.1)

    with col2:
        fixed_carbon = st.number_input("Fixed Carbon (%)", min_value=0.0, max_value=90.0, value=48.0, step=0.1)
        sulphur = st.number_input("Sulphur (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        temperature = st.number_input("Process / Sensor Temperature", min_value=0.0, max_value=2000.0, value=1250.0, step=1.0)
        depth = st.number_input("Mining Depth", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)

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
        recommendation = get_recommendation(predicted_gcv, ash, moisture, sulphur)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted GCV", f"{predicted_gcv} kcal/kg")
        c2.metric("Coal Grade", grade)
        c3.metric("Quality", quality)
        c4.metric("Pricing", price)

        st.success(recommendation)


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


def sensor_simulation_page(df, model):
    st.subheader("Sensor Fusion Simulation")
    st.write(
        """
        This page simulates real-time sensor/sample input. In a real mine, these values can come from
        IoT sensors, lab automation systems, or mine dispatch APIs.
        """
    )

    if st.button("Generate Live Sensor Sample", type="primary"):
        sample = {
            "mine_location": random.choice(df["mine_location"].unique().tolist()),
            "moisture": round(random.uniform(3, 12), 2),
            "ash": round(random.uniform(5, 25), 2),
            "volatile_matter": round(random.uniform(18, 42), 2),
            "sulphur": round(random.uniform(0.4, 3.8), 2),
            "temperature": round(random.uniform(1100, 1480), 1),
            "depth": round(random.uniform(20, 310), 1),
        }
        sample["fixed_carbon"] = round(
            100 - sample["moisture"] - sample["ash"] - sample["volatile_matter"], 2
        )

        predicted_gcv = predict_single(model, sample)
        grade = get_grade(predicted_gcv)

        st.json(sample)

        c1, c2, c3 = st.columns(3)
        c1.metric("Live Predicted GCV", f"{predicted_gcv} kcal/kg")
        c2.metric("Predicted Grade", grade)
        c3.metric("Quality", get_quality_category(predicted_gcv))

        st.success(
            get_recommendation(
                predicted_gcv,
                sample["ash"],
                sample["moisture"],
                sample["sulphur"],
            )
        )


def main():
    show_header()

    df = load_data()
    model = load_model()

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Single Prediction",
            "CSV Batch Prediction",
            "Analytics",
            "Sensor Simulation",
        ],
    )

    if page == "Home":
        home_page(df, metrics)
    elif page == "Single Prediction":
        single_prediction_page(df, model)
    elif page == "CSV Batch Prediction":
        batch_prediction_page(model)
    elif page == "Analytics":
        analytics_page(df, model)
    elif page == "Sensor Simulation":
        sensor_simulation_page(df, model)


if __name__ == "__main__":
    main()
