# House Price Prediction Using Linear Regression

## Overview

This project implements a Machine Learning model using Linear Regression to predict house prices based on:

* Area
* Number of Bedrooms
* Number of Bathrooms

The model is trained using a housing dataset and evaluated using R² Score and Mean Squared Error (MSE).

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

## Dataset Features

* Area
* Bedrooms
* Bathrooms
* Price (Target Variable)

## Project Workflow

1. Load and explore the dataset
2. Select features and target variable
3. Split data into training and testing sets
4. Train Linear Regression model
5. Predict house prices
6. Evaluate model performance
7. Visualize results using graphs

## Model Evaluation

* R² Score
* Mean Squared Error (MSE)

## Generated Visualizations

* Area vs House Price
* Bedrooms vs House Price
* Bathrooms vs House Price
* Actual vs Predicted House Prices

## Project Structure

HousePricePrediction/
├── house_price.csv
├── linear_regression.py
├── graphs/
│   ├── area_vs_price.png
│   ├── bedrooms_vs_price.png
│   ├── bathrooms_vs_price.png
│   └── actual_vs_predicted.png
└── README.md

## How to Run

1. Clone the repository
2. Install dependencies:

pip install pandas numpy scikit-learn matplotlib

3. Run the project:

python linear_regression.py

## Learning Outcomes

This project demonstrates:

* Data preprocessing
* Feature selection
* Linear Regression modeling
* Model evaluation
* Data visualization

## Author

Developed as part of a Machine Learning internship task.
