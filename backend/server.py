
import os
import sys
import json
import subprocess

import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from plotly.io import to_html
from flask import Flask, jsonify, request, render_template


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static")
)


# --------------------------------------------------
# Run backend Python scripts safely
# --------------------------------------------------

def run_script(script_name):
    script_path = os.path.join(BACKEND_DIR, f"{script_name}.py")

    if not os.path.isfile(script_path):
        return {
            "output": {},
            "error": f"Script not found: {script_path}"
        }

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=BASE_DIR
        )

        if result.returncode != 0:
            return {
                "output": {},
                "error": (
                    f"{script_name} failed with exit code "
                    f"{result.returncode}.\n"
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            }

        try:
            output_json = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            return {
                "output": {},
                "error": (
                    f"{script_name} did not return valid JSON. "
                    f"Parsing error: {exc}. "
                    f"Output: {result.stdout[:1000]}"
                )
            }

        return {
            "output": output_json,
            "error": ""
        }

    except subprocess.TimeoutExpired:
        return {
            "output": {},
            "error": f"{script_name} timed out after 120 seconds."
        }

    except Exception as exc:
        return {
            "output": {},
            "error": f"{script_name}: {str(exc)}"
        }


# --------------------------------------------------
# Main pages
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("landing.html")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


# --------------------------------------------------
# Individual analysis endpoints
# --------------------------------------------------

@app.route("/run_analysis", methods=["GET"])
def run_analysis():
    result = run_script("data_analysis")

    status = 500 if result["error"] else 200

    return jsonify({
        "output": result["output"],
        "error": result["error"]
    }), status


@app.route("/run_ml", methods=["GET"])
def run_ml():
    result = run_script("ml_model")

    status = 500 if result["error"] else 200

    return jsonify({
        "output": result["output"],
        "error": result["error"]
    }), status


@app.route("/run_recommendations", methods=["GET"])
def run_recommendations():
    result = run_script("recommendations")

    status = 500 if result["error"] else 200

    return jsonify({
        "output": result["output"],
        "error": result["error"]
    }), status


# --------------------------------------------------
# Run analysis, ML model and recommendations
# --------------------------------------------------

@app.route("/run_all", methods=["GET"])
def run_all():
    analysis = run_script("data_analysis")
    ml = run_script("ml_model")
    recommendations = run_script("recommendations")

    results = {
        "data_analysis": analysis,
        "ml_model": ml,
        "recommendations": recommendations
    }

    errors = {
        name: result["error"]
        for name, result in results.items()
        if result["error"]
    }

    response = {
        "data_analysis": analysis["output"],
        "ml_model": ml["output"],
        "recommendations": recommendations["output"],
        "errors": errors
    }

    # Return HTTP 500 if any backend script failed.
    status = 500 if errors else 200

    return jsonify(response), status


# --------------------------------------------------
# Dataset overview and visualizations
# --------------------------------------------------

@app.route("/dataset_overview")
def dataset_overview():
    csv_path = os.path.join(BASE_DIR, "food_wastage_data.csv")

    if not os.path.isfile(csv_path):
        return (
            f"Dataset not found: {csv_path}",
            404
        )

    try:
        df = pd.read_csv(csv_path)

        shape_info = (
            f"The dataset contains {df.shape[0]} rows "
            f"and {df.shape[1]} columns."
        )

        # 1. Data types chart
        data_types = df.dtypes.value_counts().reset_index()
        data_types.columns = ["Data Type", "Count"]
        data_types["Data Type"] = data_types["Data Type"].astype(str)

        fig_data_types = px.bar(
            data_types,
            x="Data Type",
            y="Count",
            title="Column Data Types"
        )

        html_data_types = to_html(
            fig_data_types,
            full_html=False,
            include_plotlyjs="cdn"
        )

        # 2. Missing values chart
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        html_missing = None

        if not missing.empty:
            fig_missing = px.bar(
                x=missing.index,
                y=missing.values,
                title="Missing Values by Column",
                labels={
                    "x": "Column",
                    "y": "Missing Values"
                }
            )

            html_missing = to_html(
                fig_missing,
                full_html=False,
                include_plotlyjs=False
            )

        # 3. Categorical distribution charts
        category_plots = []

        categorical_columns = [
            "Event_Type",
            "Storage_Conditions",
            "Type_of_Food",
            "Geographical_Location"
        ]

        for col in categorical_columns:
            if col in df.columns:
                dist = df[col].value_counts().reset_index()
                dist.columns = [col, "Count"]

                fig = px.bar(
                    dist,
                    x=col,
                    y="Count",
                    title=f"Distribution of {col}"
                )

                category_plots.append(
                    to_html(
                        fig,
                        full_html=False,
                        include_plotlyjs=False
                    )
                )

        # 4. Average waste by event type
        html_avg_waste = None

        if (
            "Event_Type" in df.columns
            and "Wastage_Food_Amount" in df.columns
        ):
            avg_waste = (
                df.groupby("Event_Type")["Wastage_Food_Amount"]
                .mean()
                .reset_index()
            )

            fig_avg = px.bar(
                avg_waste,
                x="Event_Type",
                y="Wastage_Food_Amount",
                title="Average Wastage per Event Type",
                labels={
                    "Wastage_Food_Amount": "Average Waste (kg)"
                }
            )

            html_avg_waste = to_html(
                fig_avg,
                full_html=False,
                include_plotlyjs=False
            )

        return render_template(
            "dataset_overview.html",
            shape_info=shape_info,
            html_data_types=html_data_types,
            html_missing=html_missing,
            category_plots=category_plots,
            html_avg_waste=html_avg_waste
        )

    except Exception as exc:
        app.logger.exception("Dataset overview failed")
        return f"Unable to generate dataset overview: {exc}", 500


# --------------------------------------------------
# Food waste prediction
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    recommendations = []

    if request.method == "POST":
        try:
            guests = int(request.form["guests"])
            event_type = request.form["event_type"]
            storage = request.form["storage"]

            if guests < 0:
                raise ValueError("Number of guests cannot be negative.")

            df_input = pd.DataFrame([{
                "Number_of_Guests": guests,
                "Event_Type": event_type,
                "Storage_Conditions": storage
            }])

            # Load model artifacts from the project root
            columns_path = os.path.join(BASE_DIR, "model_columns.pkl")
            scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
            model_path = os.path.join(BASE_DIR, "ml_model.pkl")

            for path in (columns_path, scaler_path, model_path):
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Required model file not found: {path}"
                    )

            expected_cols = joblib.load(columns_path)

            df_encoded = pd.get_dummies(df_input)

            for col in expected_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            df_encoded = df_encoded[expected_cols]

            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(df_encoded)

            model = joblib.load(model_path)
            prediction = round(float(model.predict(X_scaled)[0]), 2)

            # Rule-based recommendations
            if guests > 100:
                recommendations.append(
                    "Consider portion control for large guest counts."
                )

            if event_type == "Wedding":
                recommendations.append(
                    "Consider donating surplus food after the event."
                )

            if storage == "Ambient":
                recommendations.append(
                    "Use suitable cold storage where possible to reduce spoilage."
                )

            if prediction > 30:
                recommendations.append(
                    "Expected waste is high. Review menu quantities and planning."
                )

        except Exception as exc:
            app.logger.exception("Prediction failed")
            return f"Unable to generate prediction: {exc}", 500

    return render_template(
        "predict.html",
        prediction=prediction,
        recommendations=recommendations
    )


# --------------------------------------------------
# Start Flask server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
