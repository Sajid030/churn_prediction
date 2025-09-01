from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Categorical column that were mapped
mapping_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Categorical columns that were one-hot encoded
one_hot_columns = [
    'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect raw inputs from the form
        form_data = request.form.to_dict()

        # Convert to DataFrame for consistency
        input_df = pd.DataFrame([form_data])

        # Convert correct dtypes
        input_df['tenure'] = input_df['tenure'].astype(int)
        input_df['MonthlyCharges'] = input_df['MonthlyCharges'].astype(float)
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)
        input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)

        # Mapping some columns
        for col in mapping_columns:
            input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})

        # One-hot encoding (same as training)
        input_df = pd.get_dummies(input_df, columns=one_hot_columns, drop_first=True)

        # Align with training columns (fill missing with 0)
        # Load column names from training (saved in a txt or pickle during training)
        with open("columns.pkl", "rb") as f:
            train_columns = pickle.load(f)

        input_df = input_df.reindex(columns=train_columns, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template("result.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
