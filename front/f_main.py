import streamlit as st

import requests
import urllib
import json
import os
import numpy as np
import pandas as pd
import urllib.request
import urllib.request
import base64








def set_background(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
def is_docker():
    try:
        with open('/proc/1/cgroup', 'rt') as ifh:
            return 'docker' in ifh.read()
    except FileNotFoundError:
        return False

columns = ['ID','Sender_IP','Sender_Port','Target_IP','Target_Port',
               'Transport_Protocol','Duration','AvgDuration','PBS','AvgPBS','TBS',
               'PBR','AvgPBR','TBR','Missed_Bytes','Packets_Sent','Packets_Received','SRPR']





if is_docker():
    # If we're running in a Docker container, use the service name
    BACKEND_URL = "http://fastapi:8000"
else:
    
    
    BACKEND_URL = "http://localhost:8000"


MODELS_URL = urllib.parse.urljoin(BACKEND_URL, "models")
TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "train")
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")
FI_URL = urllib.parse.urljoin(BACKEND_URL, "importance")

st.set_page_config(layout="centered",
                   page_icon="💰",
                   page_title="HR Analytics - Employee Attrition Prediction")

page = "Predict"
# Change the color of the sidebar to blue



    




        

if page == "Predict":
    st.markdown("<h1 style='text-align: center; color: white;'>Hello there! 👋 Predict and check botnet</h1>", unsafe_allow_html=True)

  
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
          
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")
    
    c1, c2, c3 = st.columns(3)
    
    x1 = c1.text_input("Sender IP:", value='255.255.255.255')
    x2 = c1.number_input("Sender Port:", min_value=0, max_value=65535, value=0)
    x3 = c1.text_input("Target IP:", value='255.255.255.255')
    x4 = c1.number_input("Target Port:", min_value=0, max_value=65535, value=0)
    x5 = c1.selectbox("Transport Protocol:", options=['TCP', 'UDP'])
    x6 = c1.number_input("Duration:", value=0)

    x7 = c2.number_input("Average Duration:", value=0)
    x8 = c2.number_input("PBS:", value=0)
    x9 = c2.number_input("Average PBS:", value=0)
    x10 = c2.number_input("TBS:", value=0)
    x11 = c2.number_input("PBR:", value=0)
    x12 = c2.number_input("Average PBR:", value=0)

    x13 = c3.number_input("TBR:", value=0)
    x14 = c3.number_input("Missed Bytes:", value=0)
    x15 = c3.number_input("Packets Sent:", value=0)
    x16 = c3.number_input("Packets Received:", value=0)
    x17 = c3.number_input("SRPR:", value=0)

        
    df = (x1, x2, x3, x4, 1, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17)
    print(df)


    if st.button("Predict"):
        print(PREDICT_URL)
        try:
            response_predict = requests.post(url=PREDICT_URL,
                                              data=json.dumps({"data": df, "model_name": model_name})
                                              )
            print("Response sent!")
            print(response_predict)
            if response_predict.ok:
                print(response_predict.status_code)
                print("Response Code:", response_predict.status_code)
                res = response_predict.json()
                st.markdown(f"**Prediction**: {res['result']}")
                
            else:
                print("Some error occured!")
                st.write("Some error occured")
        except ConnectionError as e:
            st.write("Couldn't reach backend")
elif page == "Feature Importance":
    st.header("Feature Importance")
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")

    fi = requests.post(FI_URL,  data=json.dumps({"model_name": model_name}))
    fi = pd.DataFrame(fi.json()).reset_index(drop =True )
    st.dataframe(fi, 600, 600)
    
else:
    st.write("Page does not exist")
    



set_background('assets/IOT.jpg')


