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
dirs="D:\\Projects\\Freelance\\plainpythontoflask\\py-PredictiveMaintenance\\asset_information_template.xlsx"
rul=pd.read_excel(dirs,sheet_name='RUL')
failure_data=pd.read_excel(dirs,sheet_name='Failure')
service_record=pd.read_excel(dirs,sheet_name='Service Record')

sensor_data=rul.sample(n=1000)
del sensor_data['Asset']

############ solution 1  ----- 
def get_pca_graph(sensor_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sensor_data.sample(n=1000))
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    print(pca_data)
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
    
    fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
    ax.set_ylim(-2, 1.75)
    ax.set_xlim(min_date, max_date)
    ax.axhline(0, xmin=0, xmax=1, c='deeppink', zorder=1)
    ax.scatter(dates, np.zeros(len(dates)), s=120, c='palevioletred', zorder=2)
    ax.scatter(dates, np.zeros(len(dates)), s=30, c='darkmagenta', zorder=3)
    label_offsets = np.zeros(len(dates))
    label_offsets[::2] = 0.35
    label_offsets[1::2] = -0.7
    for i, (l, d) in enumerate(zip(labels, dates)):
        _ = ax.text(d, label_offsets[i], l, ha='center', fontfamily='serif', 
                    fontweight='bold', color='royalblue',fontsize=9)
        
    stems = np.zeros(len(dates))
    stems[::2] = 0.3
    stems[1::2] = -0.3
    dataset = pd.DataFrame({'dates': dates, 'stems': stems})
    print(dataset)
    # markerline, stemline, baseline = ax.stem(dates, stems, use_line_collection = True)
    # markerline, stemline, baseline = plt.stem(dates, stems, linefmt ='grey', markerfmt ='D', bottom = 1.1)
    # print(markerline)
    # print(stemline)
    # print(baseline)
    # markerline.set_markerfacecolor('darkmagenta')
    # baseline.set_markerfacecolor('darkmagenta')
    # plt.setp(markerline, marker=',', color='darkmagenta')
    # plt.setp(stemline, color='darkmagenta')
    
    # for spine in ["left", "top", "right", "bottom"]:
        # _ = plt.spines[spine].set_visible(False)
    # plt.show()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # return ax.set_title('Asset Maintenance Timeline', fontweight="bold", fontfamily='serif', fontsize=16, 
                 # color='green')

def plot_failure_correlation_data(failure_data):
    del failure_data['Asset']
    print(failure_data)
    return failure_data

# generate_pca_graph(sensor_data)
# failure_prediction_accuracy(failure_data)
# plot_failure_correlation(failure_data)
future_Service_timeline(service_record)
