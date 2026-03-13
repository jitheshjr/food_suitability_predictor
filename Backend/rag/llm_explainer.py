import subprocess
from .rag_retriever import retrieve_context

def generate_explanation(user_data, food, prediction):
    query = f"Is {food} good for {user_data['Disease']} diet"
    context = retrieve_context(query)

    prompt = f"""
        A dietary recommendation system predicted that the food '{food}' is '{prediction}'.

        User details:
        Age: {user_data['Age']}
        BMI: {user_data['BMI']}
        Disease: {user_data['Disease']}

        Food nutrition highlights:
        Calories: {user_data['Calories']}
        Sugar: {user_data['Sugar']}
        Fiber: {user_data['Fiber']}
        Sodium: {user_data['Sodium']}

        Medical knowledge:
        {context}

        Explain clearly why this recommendation is correct for the user.
        """
    result = subprocess.run(
        ["ollama", "run", "phi3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        return "Explanation generation failed."

    return result.stdout.strip()