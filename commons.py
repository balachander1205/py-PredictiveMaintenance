import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import date
import seaborn as sns
# #Read Data 
# dirs="D:\\Projects\\Freelance\\plainpythontoflask\\py-PredictiveMaintenance\\asset_information_template.xlsx"
# rul=pd.read_excel(dirs,sheet_name='RUL')
# failure_data=pd.read_excel(dirs,sheet_name='Failure')
# service_record=pd.read_excel(dirs,sheet_name='Service Record')
# sensor_data=rul.sample(n=1000)
# del sensor_data['Asset']

############ solution 1  ----- 
def get_pca_graph(sensor_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sensor_data.sample(n=1000))
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    # Extract principal components for visualization (assuming you named them PC1 and PC2)
    PC1 = pca_data[:, 0]
    PC2 = pca_data[:, 1]
    return PC1, PC2

### Classify Failure
def get_failure_prediction_accuracy(failure_data):
    # Split data into features (sensor readings) and target (failure flag)
    X = failure_data[['Temperature (C)', 'Pressure (psi)', 'Vibration (mm/s)']]
    y = failure_data['Failure Flag']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

############ solution 2  ----- 
def future_Service_timeline(service_record):
    ##Service record - Maintenance suggestion
    service_record['Date']=pd.to_datetime(service_record['Date'],format='%YYY-MM-DD')
    date_diffs = service_record['Date'].diff()
    avg_diff = date_diffs.mean().days  # Convert to days
    max_date = service_record['Date'].max()
    cc=0
    for ii in range(1,5):
        #k=k+avg
        cc=cc+1
        #service_record=pd.concat([service_record,pd.DataFrame({'Date':[k.strftime('%Y-%m-%d')],'Type':['Future Maintenance '+str(cc)]})])
        new_date = max_date + pd.Timedelta(days=avg_diff*ii)
        new_row = {'Date': new_date,'Type':'Future Maintenance '+str(cc)}
        service_record = service_record._append(new_row, ignore_index=True)  # Append at the end
    
    max_date = service_record['Date'].max()
    min_date = service_record['Date'].min()
    
    labels = service_record.Type
    dates = service_record.Date
    # labels with associated dates
    labels = ['{0:%d %b %Y}:\n{1}'.format(d, l) for l, d in zip (labels, dates)]
    label_offsets = np.zeros(len(dates))
    label_offsets[::2] = 0.35
    label_offsets[1::2] = -0.7
        
    stems = np.zeros(len(dates))
    stems[::2] = 0.3
    stems[1::2] = -0.3
    dataset = pd.DataFrame({'date': dates, 'stems': stems})
    return dataset

def plot_failure_correlation_data(failure_data):
    del failure_data['Asset']
    return failure_data
