import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "ml", "rf_model.pkl")
columns_path = os.path.join(BASE_DIR, "ml", "model_columns.pkl")
dataset_path = os.path.join(BASE_DIR, "dataset", "lookup_dataset.csv")

model = joblib.load(model_path)
feature_columns = joblib.load(columns_path)
nutrition_data = pd.read_csv(dataset_path)

nutrition_data["Food_Name"] = nutrition_data["Food_Name"].str.lower()
nutrition_data.set_index("Food_Name", inplace=True)

def calculate_bmi(height, weight):

    height_m = height / 100
    bmi = weight / (height_m ** 2)

    return round(bmi, 2)

def calculate_bmr(weight, height, age, gender):

    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    return round(bmr, 2)

def get_food_nutrition(food_name):

    food_name = food_name.lower()

    if food_name not in nutrition_data.index:
        return None

    return nutrition_data.loc[food_name].to_dict()

def predict_food_suitability(input_data):

    df = pd.DataFrame([input_data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)[0]

    label_map = {
        0: "Suitable",
        1: "Moderate",
        2: "Not Suitable"
    }

    return label_map.get(prediction, "Unknown")