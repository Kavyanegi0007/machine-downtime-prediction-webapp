# Import necessary libraries
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Step 1: Generate Synthetic Data (with new features)
np.random.seed(42)

n_samples = 1000
data = pd.DataFrame({
    'Machine_ID': np.arange(1, n_samples + 1),
    'Temperature': np.random.uniform(60, 100, n_samples),  # Simulate machine temperature
    'Run_Time': np.random.uniform(50, 300, n_samples),  # Simulate runtime
    'Machine_Age': np.random.uniform(1, 10, n_samples),  # Simulate machine age in years
    'Humidity': np.random.uniform(30, 80, n_samples),  # Simulate humidity as a percentage
    'Downtime_Flag': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # Random downtime
})

# Step 2: Prepare data for training
X = data[['Temperature', 'Run_Time', 'Machine_Age', 'Humidity']]  # Features
y = data['Downtime_Flag']  # Target

# Step 3: Normalize the data (important for LSTM and SMOTE)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(X)

# Step 4: Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(scaled_data, y)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 6: Define and train the Logistic Regression model with class weights
model = LogisticRegression(class_weight='balanced')  # Use class weights to handle imbalanced data
model.fit(X_train, y_train)

# Define endpoints
@app.route('/')
def home():
    return "Welcome to the Machine Downtime Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json(force=True)
        temp = data['Temperature']
        run_time = data['Run_Time']
        machine_age = data['Machine_Age']
        humidity = data['Humidity']
        
        # Prepare the input data (normalize it)
        input_data = np.array([[temp, run_time, machine_age, humidity]])
        input_data_scaled = scaler.transform(input_data)  # Normalize the input data
        
        # Make predictions and get probabilities
        prediction = model.predict(input_data_scaled)
        predicted_downtime = 'Yes' if prediction[0] == 1 else 'No'
        confidence = model.predict_proba(input_data_scaled)[0][prediction[0]]  # Get the confidence of the prediction
        
        # Respond with prediction and confidence
        return jsonify({
            'Predicted downtime': predicted_downtime,
            'Confidence': round(confidence, 2)  # Rounded to 2 decimal places
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(port=5000)
