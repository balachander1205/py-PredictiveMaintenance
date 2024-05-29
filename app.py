import logging
from flask import Flask,render_template, request,json,Response, jsonify

from commons import future_Service_timeline, get_pca_graph, get_failure_prediction_accuracy, plot_failure_correlation_data
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import json
from flask_swagger_ui import get_swaggerui_blueprint
import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
import os
import pandas as pd
from werkzeug.serving import run_simple

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './static/uploads/'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# logging config
handler = RotatingFileHandler('static/logs/middleware.log', maxBytes=1)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")
handler.setFormatter(formatter)
logging.getLogger('').setLevel(logging.DEBUG)
logging.getLogger('').addHandler(handler)

# swagger config
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config = {
        'app_name' : 'customer review sentiment analysis Application'
    }
)
app.register_blueprint(SWAGGER_BLUEPRINT, url_prefix = SWAGGER_URL)


@app.route('/')
@cross_origin()
def hello():
    return 'Welcome to image validation Application!'

@app.route('/home')
@cross_origin()
def parkingManagement():
    return render_template('index.html')

@app.route('/validateImage', methods=['POST'])    
@cross_origin()
def validateImage():
    filePath = request.args.get('filePath', '')
    return json.dumps({'status':'OK','imagesrc':'', 'data':'', 'date':'', 'vehcount':'', 'qrcount':'', 'tabledata':''});

@app.route('/getPcaGraph', methods=['GET','POST'])    
@cross_origin()
def get_pca_graph_data():
    try:
        global sensor_data
        if 'sensor_data' not in globals():
            return jsonify({"message": "No CSV file uploaded"}), 400

        pc1, pc2, base64_str = get_pca_graph(sensor_data)
        data = {
          "green": pc1.tolist(),
          "red": pc2.tolist(),
          "image": base64_str
        }
        return Response(json.dumps(data),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:get_pca_graph_data="+e)


@app.route('/getFPAccuracy', methods=['GET','POST'])    
@cross_origin()
def failure_prediction_accuracy_data():
    try:
        global failure_data
        if 'failure_data' not in globals():
            return jsonify({"message": "No CSV file uploaded"}), 400

        accuracy, base64_str = get_failure_prediction_accuracy(failure_data)
        data = {
          "accuracy": accuracy,
          "image": base64_str
        }
        return Response(json.dumps(data),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:failure_prediction_accuracy_data="+e)

@app.route('/getPlotFailure', methods=['GET','POST'])    
@cross_origin()
def get_plot_failure_correlation():
    try:
        global failure_data
        if 'failure_data' not in globals():
            return jsonify({"message": "No CSV file uploaded"}), 400
        data_frame, base64_str = plot_failure_correlation_data(failure_data)
        failure_data1 = json.loads(data_frame.to_json(orient="records"))
        data = {
            "dataset": failure_data1,
            "image": base64_str
        }
        return Response(json.dumps(data),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:get_plot_failure_correlation="+e)


@app.route('/getFSTimeline', methods=['GET','POST'])    
@cross_origin()
def get_future_Service_timeline():
    try:
        global service_record
        if 'service_record' not in globals():
            return jsonify({"message": "No CSV file uploaded"}), 400

        service_record_df = service_record
        dataset, base64_str = future_Service_timeline(service_record_df)
        dataframe = json.loads(dataset.to_json(orient="records"))
        data = {
            "dataset": dataframe,
            "image":base64_str
        }
        return Response(json.dumps(data),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:get_future_Service_timeline="+e)

@app.route('/upload', methods=['GET','POST'])
@cross_origin()
def upload_csv():
    file = request.files['file']
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        try:
            # Read the File using Flask request
            # file = request.files['file']
            filename = secure_filename(file.filename)
            dirs = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dirs)
            global rul
            global failure_data
            global service_record
            global sensor_data
            
            rul = pd.read_excel(dirs,sheet_name='RUL')
            failure_data = pd.read_excel(dirs,sheet_name='Failure')
            service_record = pd.read_excel(dirs,sheet_name='Service Record')
            sensor_data = rul.sample(n=1000)
            del sensor_data['Asset']
            
            return jsonify({"message": "File uploaded successfully"})
        except Exception as e:
            return jsonify({"message": str(e)}), 500

if __name__=="__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)    
    logging.info('Started')
    # app.run(host='127.0.0.1', port=5001,debug=True, threaded=True)
    run_simple('127.0.0.1', 5001, app)
