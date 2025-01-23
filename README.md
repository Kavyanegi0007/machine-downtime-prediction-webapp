

# Machine Downtime Prediction API

This API predicts machine downtime based on various factors such as temperature, run time, machine age, and humidity using a Logistic Regression model. It uses synthetic data and is designed for use in machine maintenance and predictive analytics.

## Prerequisites

Before running the Flask app, ensure that you have the following installed on your Windows PC:

1. **Python 3.8+**
2. **pip (Python package installer)**

### Required Libraries

Install the required libraries by running the following command in your command prompt or terminal:

```bash
pip install pandas numpy scikit-learn imbalanced-learn flask


example curl request:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"Temperature\": 85.5, \"Run_Time\": 150, \"Machine_Age\": 5, \"Humidity\": 60}"
example output:
{
    "Predicted downtime": "Yes",
    "Confidence": 0.75
}
