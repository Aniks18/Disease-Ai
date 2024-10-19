from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('./improved_epidemic_model_nn_v2.keras')

# Define disease groups
disease_groups = ['Viral Infections', 'Bacterial Infections', 'Cancer', 'Syndromes', 'Chronic Diseases', 'Disorders']

app = Flask(__name__)

def encode_input(inputs):
    """Simple encoding method - adjust based on your model's expectations."""
    encoded = []
    for input in inputs:
        if isinstance(input, str):
            encoded.append(hash(input.lower()) % 10)  # Convert string to a number between 0-9
        elif isinstance(input, (int, float)):
            encoded.append(min(int(input / 10), 9))  # Convert number to a single digit, max 9
    return encoded

def predict_disease(inputs):
    """Predict disease based on encoded inputs."""
    # Encode the inputs
    encoded_inputs = encode_input(inputs)
    
    # Ensure we have 10 inputs as expected by the model
    while len(encoded_inputs) < 10:
        encoded_inputs.append(0)  # Pad with zeros if needed
    
    # Convert to numpy array and reshape
    input_array = np.array(encoded_inputs[:10]).reshape(1, -1)
    
    # Make prediction
    prediction_proba = model.predict(input_array)
    prediction = np.argmax(prediction_proba, axis=1)
    
    return disease_groups[prediction[0]], prediction_proba[0]

@app.route('/predict', methods=['GET'])
def predict():
    symptom1 = request.args.get('symptom1')
    symptom2 = request.args.get('symptom2')
    symptom3 = request.args.get('symptom3', '')  # Optional parameter
    age = int(request.args.get('age'))
    bmi = float(request.args.get('bmi'))
    temperature = float(request.args.get('temperature'))
    season = request.args.get('season')

    user_inputs = [symptom1, symptom2, symptom3, age, bmi, temperature, season]
    
    predicted_disease, probabilities = predict_disease(user_inputs)

    # Convert probabilities from numpy float32 to native Python float
    probabilities = {disease: float(prob) for disease, prob in zip(disease_groups, probabilities)}

    return jsonify({
        "predicted_disease": predicted_disease,
        "probabilities": probabilities
    })

if __name__ == "__main__":
    app.run(debug=True)
