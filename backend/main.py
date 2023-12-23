import numpy as np
from fastapi import FastAPI
from fastapi import BackgroundTasks
from urllib.parse import urlparse
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
import sys
from ml.data import  preprocessing, config
remote_server_uri="https://dagshub.com/medmabcf/N-BaIoT_mlops.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
mlflowclient = MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())
"""df_mlflow = mlflow.search_runs(filter_string="metrics.f1_score<1")
run_id = df_mlflow.loc[df_mlflow['metrics.f1_score'].idxmax()]['run_id']
runname = df_mlflow.loc[df_mlflow['metrics.f1_score'].idxmax()]['tags.mlflow.runName']    """


app = FastAPI()

#from backend.models import  PredictApiData


cols = config().columns

@app.get("/best")
async def get_models_api():
    models = mlflowclient.search_registered_models()
   
    best_model = None
    best_f1_score = -np.inf
    
    for model in models:
        for version in mlflowclient.get_latest_versions(model.name):
            run = mlflowclient.get_run(version.run_id)
            f1_score = run.data.metrics.get('f1')
            if f1_score is not None and f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = version
    print(best_model.name)
    return [best_model.name]

@app.get("/models")
async def get_models_api():
    models = mlflowclient.search_registered_models()
   
    models_list = []
    
    for model in models:
        models_list.append(model.name)
    return models_list




