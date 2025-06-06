from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Create route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') # Displays simple data fields we need to provide to our model
    
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_enthnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print(results)
        return render_template('home.html', results=results[0]) # Display the prediction result on the same page
    

if __name__ == "__main__":
    #app.run(debug=True)  # Run the Flask app in debug mode for development purposes
    app.run(host='0.0.0.0', debug=True)  # Run the Flask app on all interfaces at port 5000