from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained XGBoost model
model = pickle.load(open("bigmart_model.pkl", "rb"))

# Define the mapping for categorical variables (use same encoding as training)
item_type_mapping = {
    "Beverages": 0,
    "Snack Foods": 1,
    "Dairy": 2,
    "Frozen Foods": 3,
    # ... add all your Item_Type mappings
}

outlet_type_mapping = {
    "Supermarket Type1": 0,
    "Supermarket Type2": 1,
    "Grocery Store": 2,
    # ... add all your Outlet_Type mappings
}

outlet_location_mapping = {
    "Tier 1": 0,
    "Tier 2": 1,
    "Tier 3": 2
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    item_mrp = float(request.form['item_mrp'])
    item_weight = float(request.form['item_weight'])
    item_visibility = float(request.form['item_visibility'])
    item_type = request.form['item_type']
    outlet_type = request.form['outlet_type']
    outlet_location = request.form['outlet_location']

    # Encode categorical variables
    item_type_enc = item_type_mapping[item_type]
    outlet_type_enc = outlet_type_mapping[outlet_type]
    outlet_location_enc = outlet_location_mapping[outlet_location]

    # Create feature array in the correct order
    features = np.array([[item_weight, item_visibility, item_mrp,
                          item_type_enc, outlet_type_enc, outlet_location_enc]])

    # Predict sales
    prediction = model.predict(features)[0]

    return render_template("result.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
