import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



application = Flask(__name__)
app = application

@app.route('/')
def index():
    """
    Renders the 'index.html' template when the root URL is accessed.

    :return: The rendered 'index.html' template.
    """
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    """
    Route for predicting housing prices based on input data.

    Args:
        None

    Returns:
        - If the request method is 'GET':
            - Renders the 'home.html' template.
        - If the request method is 'POST':
            - Parses input data from the request form.
            - Creates a CustomData object with the parsed data.
            - Converts the CustomData object to a pandas DataFrame.
            - Creates a PredictPipeline object.
            - Calls the predict method of the PredictPipeline object with the DataFrame.
            - Renders the 'home.html' template with the prediction results.

    Raises:
        None
    """
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            area=float(request.form.get('area')),
            bedrooms=int(request.form.get('bedrooms')),
            bathrooms=int(request.form.get('bathrooms')),
            stories=int(request.form.get('stories')),
            parking=int(request.form.get('parking')),
            mainroad=request.form.get('mainroad'),
            guestroom=request.form.get('guestroom'),
            basement=request.form.get('basement'),
            hotwaterheating=request.form.get('hotwaterheating'),
            airconditioning=request.form.get('airconditioning'),
            furnishingstatus=request.form.get('furnishingstatus'),
            prefarea=request.form.get('prefarea')
            
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = np.round(predict_pipeline.predict(pred_df),2)

        return render_template('home.html',results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)





