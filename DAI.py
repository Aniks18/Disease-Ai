import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('./improved_epidemic_model_nn_v2.keras')

# Define disease groups
disease_groups = ['Viral Infections', 'Bacterial Infections', 'Cancer', 'Syndromes', 'Chronic Diseases', 'Disorders']

def get_user_input():
    print("Please provide the following information:")
    
    symptom1 = input("Primary symptom: ")
    symptom2 = input("Secondary symptom: ")
    symptom3 = input("Tertiary symptom (if any, or press Enter to skip): ")
    age = int(input("Age: "))
    bmi = float(input("BMI: "))
    temperature = float(input("Current temperature in Celsius: "))
    season = input("Current season: ")

    return symptom1, symptom2, symptom3, age, bmi, temperature, season

def encode_input(inputs):
    # Simple encoding method - you might need to adjust this based on your model's expectations
    encoded = []
    for input in inputs:
        if isinstance(input, str):
            encoded.append(hash(input.lower()) % 10)  # Convert string to a number between 0-9
        elif isinstance(input, (int, float)):
            encoded.append(min(int(input / 10), 9))  # Convert number to a single digit, max 9
    return encoded

def predict_disease(inputs):
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

# Main execution
if __name__ == "__main__":
    print("Welcome to the Simple Symptom-Based Disease Prediction System")
    print("------------------------------------------------------------")
    
    while True:
        user_inputs = get_user_input()
        try:
            predicted_disease, probabilities = predict_disease(user_inputs)

            print("\nPrediction Results:")
            print(f"Predicted Disease Group: {predicted_disease}")
            print("\nProbabilities for each disease group:")
            for disease, prob in zip(disease_groups, probabilities):
                print(f"{disease}: {prob:.4f}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with different inputs.")

        another = input("\nWould you like to make another prediction? (yes/no): ").lower()
        if another != 'yes':
            break

    print("Thank you for using the Simple Symptom-Based Disease Prediction System!")