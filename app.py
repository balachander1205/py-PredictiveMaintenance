import logging
from flask import Flask,render_template, request,json,Response

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
        # Read the File using Flask request
        file = request.files['file']
        filename = secure_filename(file.filename)
        dirs = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dirs)
        rul = pd.read_excel(dirs,sheet_name='RUL')
        failure_data = pd.read_excel(dirs,sheet_name='Failure')
        service_record = pd.read_excel(dirs,sheet_name='Service Record')
        sensor_data = rul.sample(n=1000)
        del sensor_data['Asset']
        # print(sensor_data)
        pc1, pc2 = get_pca_graph(sensor_data)
        data = {
          "green": pc1.tolist(),
          "red": pc2.tolist()
        }
        return Response(json.dumps(data),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:get_pca_graph_data="+e)


@app.route('/getFPAccuracy', methods=['GET','POST'])    
@cross_origin()
def failure_prediction_accuracy_data():
    try:
        # Read the File using Flask request
        file = request.files['file']
        filename = secure_filename(file.filename)
        dirs = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dirs)
        rul = pd.read_excel(dirs,sheet_name='RUL')
        failure_data = pd.read_excel(dirs,sheet_name='Failure')
        service_record = pd.read_excel(dirs,sheet_name='Service Record')
        sensor_data = rul.sample(n=1000)
        del sensor_data['Asset']
        # print(sensor_data)
        accuracy = get_failure_prediction_accuracy(failure_data)
        data = {
          "accuracy": accuracy
        }
        return Response(json.dumps(data),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:failure_prediction_accuracy_data="+e)

@app.route('/getPlotFailure', methods=['GET','POST'])    
@cross_origin()
def get_plot_failure_correlation():
    try:
        # Read the File using Flask request
        file = request.files['file']
        filename = secure_filename(file.filename)
        dirs = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dirs)
        rul = pd.read_excel(dirs,sheet_name='RUL')
        failure_data = pd.read_excel(dirs,sheet_name='Failure')
        service_record = pd.read_excel(dirs,sheet_name='Service Record')
        sensor_data = rul.sample(n=1000)
        del sensor_data['Asset']
        failure_data1 = json.loads(failure_data.to_json(orient="records"))
        return Response(json.dumps(failure_data1),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:get_plot_failure_correlation="+e)


@app.route('/getFSTimeline', methods=['GET','POST'])    
@cross_origin()
def get_future_Service_timeline():
    try:
        # Read the File using Flask request
        file = request.files['file']
        filename = secure_filename(file.filename)
        dirs = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dirs)
        service_record = pd.read_excel(dirs,sheet_name='Service Record')
        dataset = future_Service_timeline(service_record)
        dataframe = json.loads(dataset.to_json(orient="records"))
        return Response(json.dumps(dataframe),mimetype='application/json')
    except Exception as e:
        print(e)
        logging.debug("Xception:get_future_Service_timeline="+e)

if __name__=="__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)    
    logging.info('Started')
    app.run(host='127.0.0.1', port=5001,debug=True, threaded=True)
