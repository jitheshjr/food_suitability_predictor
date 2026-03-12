from .agent_tools import (
    calculate_bmi,
    calculate_bmr,
    get_food_nutrition,
    predict_food_suitability
)

from rag.llm_explainer import generate_explanation


def run_agent(data):

    try:

        age = int(data["age"])
        gender = data["gender"].lower()
        height = float(data["height"])
        weight = float(data["weight"])
        disease = data["disease"].lower()
        activity_level = data["activity_level"].lower()
        food = data["food"].lower()

        bmi = calculate_bmi(height, weight)
        bmr = calculate_bmr(weight, height, age, gender)

        nutrition = get_food_nutrition(food)

        if nutrition is None:
            return {"error": "Food not found in database"}

        food_category = nutrition["Food_Category"]

        input_data = {
            "Age": age,
            "Height": height,
            "Weight": weight,
            "BMI": bmi,
            "BMR": bmr,
            "Gender": gender,
            "Disease": disease,
            "Activity_Level": activity_level,
            "Food_Category": food_category,
            "Calories": nutrition["Calories"],
            "Sugar": nutrition["Sugar"],
            "Protein": nutrition["Protein"],
            "Fat": nutrition["Fat"],
            "Carbs": nutrition["Carbohydrates"],
            "Fiber": nutrition["Fiber"],
            "Sodium": nutrition["Sodium"],
            "Cholesterol": nutrition["Cholesterol"]
        }

        prediction = predict_food_suitability(input_data)

        explanation = generate_explanation(input_data, food, prediction)

        return {
            "prediction": prediction,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}