# Machine Downtime Prediction API

## Overview
This project is a Flask-based web application that predicts machine downtime based on input parameters such as temperature, runtime, machine age, and humidity. The model is trained using synthetic data and employs a **Random Forest Classifier** for classification.

## Features
- **Web Interface**: Users can input machine-specific parameters and receive downtime predictions.
- **Machine Learning Model**: A **Random Forest Classifier** trained with **SMOTE** to handle class imbalance.
- **Feature Engineering**: Includes interaction between **Temperature** and **Run Time** as a new feature.
- **Flask API**: Accepts user input via web forms and JSON requests.
- **Data Normalization**: Uses **MinMaxScaler** for scaling features.

## Dataset
The synthetic dataset consists of the following features:
- **Temperature (°F)**: Range from **60 to 100**
- **Run Time (minutes)**: Range from **50 to 300**
- **Machine Age (years)**: Range from **1 to 10**
- **Humidity (%RH)**: Range from **30 to 80**
- **Downtime Flag**: Binary (0 = No Downtime, 1 = Downtime)
- **Temp_Run_Interaction**: Computed as `Temperature * Run_Time`

## Setup & Installation
### Prerequisites
Ensure you have **Python 3.x** installed along with the following dependencies:
```sh
pip install flask pandas numpy scikit-learn imbalanced-learn
```

### Running the Application
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/machine-downtime-prediction.git
   cd machine-downtime-prediction
   ```
2. Start the Flask server:
   ```sh
   python app.py
   ```
3. Open your browser and visit:
   ```
   http://127.0.0.1:5000/
   ```

## API Endpoints
### 1. Home Page
- **URL**: `/`
- **Method**: `GET`
- **Response**: Renders an HTML form for user input.

### 2. Predict Downtime
- **URL**: `/predict`
- **Method**: `POST`
- **Request (JSON Format)**:
  ```json
  {
    "Temperature": 85,
    "Run_Time": 200,
    "Machine_Age": 5,
    "Humidity": 50
  }
  ```
- **Response**:
  ```json
  {
    "Predicted downtime": "Yes",
    "Confidence": 0.78
  }
  ```

## Improving Confidence Scores
To get better predictions, ensure input values are within meaningful ranges:
- **Temperature**: Close to 80-90°F
- **Run Time**: Moderate range (100-250 minutes)
- **Machine Age**: Lower age (1-5 years) tends to reduce downtime risk
- **Humidity**: Stable (around 40-60%)

## Future Enhancements
- Implement **Deep Learning (LSTM)** for sequential data.
- Deploy on **AWS/GCP** for production usage.
- Improve model accuracy with **real-world data**.

## Author
- Kavya Negi  
- Email: kavyanegi0007@gmail.com

