import { useState } from "react";
import { predictFood } from "../services/api";

function FoodForm() {

  const [formData, setFormData] = useState({
    age: "",
    gender: "",
    height: "",
    weight: "",
    disease: "",
    activity_level: "",
    food_name: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await predictFood(formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>

      <h2>Food Suitability Predictor</h2>

      <form onSubmit={handleSubmit}>

        <input name="age" placeholder="Age" onChange={handleChange} />
        <input name="gender" placeholder="Gender" onChange={handleChange} />
        <input name="height" placeholder="Height" onChange={handleChange} />
        <input name="weight" placeholder="Weight" onChange={handleChange} />
        <input name="disease" placeholder="Disease" onChange={handleChange} />
        <input name="activity_level" placeholder="Activity Level" onChange={handleChange} />
        <input name="food_name" placeholder="Food Name" onChange={handleChange} />

        <button type="submit">Predict</button>

      </form>

      {result && (
        <div>
          <h3>Prediction</h3>
          <p>{result.prediction}</p>

          <h3>Explanation</h3>
          <p>{result.explanation}</p>
        </div>
      )}

    </div>
  );
}

export default FoodForm;