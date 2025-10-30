Big Mart Sales Prediction

Predict product sales for Big Mart outlets using machine learning and interactively get predictions through a Flask web app.

Project Overview

This project builds an XGBoost regression model to predict sales for different products in Big Mart stores based on historical data.
It includes a Flask web interface where users can input product and outlet details to get predicted sales dynamically, making it a full end-to-end ML + deployment project.

Features

Data cleaning and preprocessing of Big Mart dataset

Feature engineering and encoding of categorical variables

Trained XGBoost regression model with R² = 0.62 (train) / 0.61 (test)

Interactive Flask web app for predicting sales

Input form for product details (MRP, weight, visibility, type, outlet type, outlet location)

Displays predicted sales in real-time

Optional visualization of feature importance

Technologies Used

Python

Pandas & NumPy

XGBoost

Flask, HTML, CSS

Pickle for saving the trained model

Setup & Installation

Clone the repository:

git clone https://github.com/Aishwaryabagi/big_mart_sale.git
cd bigmart-sales-prediction


Install dependencies:

pip install -r requirements.txt


Make sure the trained model bigmart_model.pkl is in the project root.

Run the Flask app locally:

python app.py


Open your browser and go to:

http://127.0.0.1:5000/

Usage

Enter the product details and outlet info in the input form.

Click Predict Sales.

View predicted sales dynamically.
