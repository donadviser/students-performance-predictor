from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from pipelines.inference_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

@application.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@application.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )

        pred_df = data.prepare_data_for_inference()
        
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0],2)
        return render_template('index.html',results=results)
    
if __name__ == '__main__':
    application.run(host='0.0.0.0')