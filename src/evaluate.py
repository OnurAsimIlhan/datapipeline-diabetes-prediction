import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URL'] = ""
os.environ['MLFLOW_TRACKING_USERNAME'] = ""
os.environ['MLFLOW_TRACKING_PASSWORD'] = ""

params=yaml.safe_load(open("params.yaml"))['train']
def evaulate(data_path, model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/OnurAsimIlhan/mlpipeline.mlflow")

    ##load the model from the disk
    model=pickle.load(open(model_path, 'rb'))
    predictions=model.predict(X)
    accuracy=accuracy_score(y, predictions)

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model Accuracy: {accuracy}")

if __name__ == "__main__":
    evaulate(params["data"], params["model"])
    