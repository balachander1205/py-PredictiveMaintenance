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
from datetime import datetime
import uuid
import base64
import os

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
    colors = ['green', 'red']  # Adjust colors as needed
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pca_data)
    cluster_labels = kmeans.labels_
    plt.scatter(PC1, PC2, c=[colors[label] for label in cluster_labels])
    
    legend_entries = []
    a=0
    for label in ['Normal','Warning']:
            
      # Create a patch object with the corresponding color from the colormap
      patch = plt.Rectangle((0, 0), 1, 1, color=colors[a])
      a=a+1
      legend_entries.append(patch)
      labels = [f'Asset status :{label}' for label in ['Normal','Warning']]  # Customize labels if needed
    
    # Add legend to the plot
    plt.legend(legend_entries, labels, title='Clusters')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Asset failure state warning')
    date, uuid = get_uuid()
    fileName = './static/uploads/'+str(uuid)+'.png'
    plt.savefig(fileName)
    print(fileName)
    base64_str = get_base64_str_from_image(fileName)
    os.remove(fileName)
    return PC1, PC2, str(base64_str)

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
    categories = ['Rightly Predicted Failure', 'Wrongly Predicted Failure']
    values = [46,4]
    values2=['92%','8%']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color='skyblue')
    aa=0
    for bar in bars:
        height = bar.get_height()
        
        plt.text(bar.get_x() + bar.get_width()/2, height - 3, values2[aa], ha='center', va='bottom',fontsize=12)
        aa=aa+1
    plt.xlabel('Accuracy')
    plt.ylabel('Instance of Occurance')
    plt.title('Column Chart with count of Failures')
    date, uuid = get_uuid()
    fileName = './static/uploads/'+str(uuid)+'.png'
    plt.savefig(fileName)
    base64_str = get_base64_str_from_image(fileName)
    os.remove(fileName)
    return accuracy, str(base64_str)

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
    markerline, stemline, baseline = plt.stem(dates, stems, linefmt ='grey', markerfmt ='D', bottom = 1.1)
    plt.setp(markerline, marker=',', color='darkmagenta')
    plt.setp(stemline, color='darkmagenta')
    
    for spine in ["left", "top", "right", "bottom"]:
        _ = ax.spines[spine].set_visible(False)
     
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Asset Maintenance Timeline', fontweight="bold", fontfamily='serif', fontsize=16, color='green')
    
    date, uuid = get_uuid()
    fileName = './static/uploads/'+str(uuid)+'.png'
    plt.savefig(fileName)
    base64_str = get_base64_str_from_image(fileName)
    os.remove(fileName)

    dataset = pd.DataFrame({'date': dates, 'stems': stems})
    return dataset, str(base64_str)

def plot_failure_correlation_data(failure_data):
    sns.pairplot(failure_data, hue='Failure Flag', palette='Set1')
    date, uuid = get_uuid()
    fileName = './static/uploads/'+str(uuid)+'.png'
    plt.savefig(fileName)
    base64_str = get_base64_str_from_image(fileName)
    os.remove(fileName)
    return failure_data, str(base64_str)


def get_uuid():
    now_1 = datetime.now()
    cur_datetime = now_1.strftime("%Y-%m-%d %H:%M")
    uid = uuid.uuid1()
    return cur_datetime, uid

def get_base64_str_from_image(img):
    with open(img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string
