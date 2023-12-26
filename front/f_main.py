import streamlit as st

import requests
import urllib
import json
import os
import numpy as np
import pandas as pd
import urllib.request


columns = ['ID','Sender_IP','Sender_Port','Target_IP','Target_Port',
               'Transport_Protocol','Duration','AvgDuration','PBS','AvgPBS','TBS',
               'PBR','AvgPBR','TBR','Missed_Bytes','Packets_Sent','Packets_Received','SRPR']


local = True


if local:
    BACKEND_URL = "http://localhost:8000"
else:
    ip = urllib.request.urlopen("http://169.254.169.254/latest/meta-data/public-ipv4").read().decode()
    BACKEND_URL = str("http://") + str(ip) + str(":8000")


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
    st.header("Hello there! 👋   Predict and check  botnet " )
  
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
    
    x1 = c1.text_input("Sender IP:", value='000.000.000.000')
    x2 = c1.number_input("Sender Port:", min_value=0, max_value=65535)
    x3 = c1.text_input("Target IP:", value='000.000.000.000')
    x4 = c1.number_input("Target Port:", min_value=0, max_value=65535)
    x5 = c1.selectbox("Transport Protocol:", options=['TCP', 'UDP'])
    x6 = c1.number_input("Duration:")

    x7 = c2.number_input("Average Duration:")
    x8 = c2.number_input("PBS:")
    x9 = c2.number_input("Average PBS:")
    x10 = c2.number_input("TBS:")
    x11 = c2.number_input("PBR:")
    x12 = c2.number_input("Average PBR:")

    x13 = c3.number_input("TBR:")
    x14 = c3.number_input("Missed Bytes:")
    x15 = c3.number_input("Packets Sent:")
    x16 = c3.number_input("Packets Received:")
    x17 = c3.number_input("SRPR:")
        
    df = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17)



    if st.button("Predict"):
    
        try:
            response_predict = requests.post(url=PREDICT_URL,
                                              data=json.dumps({"data": df, "model_name": model_name})
                                              )
            print("Response sent!")
            print(response_predict)
            if response_predict.ok:
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
    

footer = """
<center>
<br>
<br>
<br>
<br>

</center>
"""

st.markdown(
    """
    <style>
        body {
            color: #fff;
            background-color: #000;
        }
        .css-17eq0hr.e1gpkc4s0 {
            background-color: #3498db;  /* Change to your desired sidebar color */
        }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(footer, unsafe_allow_html=True)
