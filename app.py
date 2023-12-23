import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import ipaddress
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import time
from sklearn.svm import SVC



df=pd.read_csv("N_BaIoT_Datasets.csv")
# Assume df is your DataFrame
df = df.drop(columns=['ID'])
df.replace(['nan', 'infinity'], np.nan, inplace=True)
df.dropna(inplace=True)
df['Sender_IP'] = df['Sender_IP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
df['Target_IP'] = df['Target_IP'].apply(lambda ip: int(ipaddress.ip_address(ip)))

scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=['class']).values
Y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

## For Remote server only(DAGShub)

remote_server_uri="https://dagshub.com/medmabcf/N-BaIoT_mlops.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
with mlflow.start_run():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    predicted_classes = knn.predict(X_test)

    precision = precision_score(y_test, predicted_classes, average='macro')
    recall = recall_score(y_test, predicted_classes, average='macro')
    f1 = f1_score(y_test, predicted_classes, average='macro')
    accuracy = accuracy_score(y_test, predicted_classes)

    print("KNN model (n_neighbors=5):")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')

    mlflow.log_param("n_neighbors", 5)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", accuracy)
    
   

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(
            knn, "model", registered_model_name="KNNBotnetModel"
        )
    else:
        mlflow.sklearn.log_model(knn, "model")
with mlflow.start_run():
    svm = SVC()
    svm.fit(X_train, y_train)

    predicted_classes = svm.predict(X_test)

    precision = precision_score(y_test, predicted_classes, average='macro')
    recall = recall_score(y_test, predicted_classes, average='macro')
    f1 = f1_score(y_test, predicted_classes, average='macro')
    accuracy = accuracy_score(y_test, predicted_classes)

    print("SVM model:")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", accuracy)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(
            svm, "model", registered_model_name="SVMBotnetModel"
        )
    else:
        mlflow.sklearn.log_model(svm, "model")
