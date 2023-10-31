from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)



# Train a simple Linear Regression model for demonstration purposes
model = pickle.load(open('R:\ML_Projects\Diamond_Price_Prediction\models\Ridge.pkl','rb'))


@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        carat = float(request.form.get('carat'))
        depth = float(request.form.get('depth'))
        table = float(request.form.get('table'))
        x = float(request.form.get('x'))
        y = float(request.form.get('y'))
        z = float(request.form.get('z'))
        cut = request.form.get('cut')
        color = request.form.get('color')
        clarity = request.form.get('clarity')
        
        # Prepare user input for prediction
        input_data = ([[carat, depth, table, x, y, z, cut, color, clarity]])
        
        # Make a price prediction
        predicted_price = model.predict(input_data)[0]
        
        return render_template('index.html', prediction=f"Predicted Price: ${predicted_price:.2f}")
    
    else:
        return render_template('index.html', prediction='')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
