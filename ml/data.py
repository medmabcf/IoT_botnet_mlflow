import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import ipaddress
from sklearn.model_selection import train_test_split




class config:
    columns = ['Sender_IP','Sender_Port','Target_IP','Target_Port',
               'Transport_Protocol','Duration','AvgDuration','PBS','AvgPBS','TBS',
               'PBR','AvgPBR','TBR','Missed_Bytes','Packets_Sent','Packets_Received','SRPR']
    
def preprocessing(df):
  # Drop the 'ID' column from the dataframe
  #df = df.drop(columns=['ID'])

  # Replace the strings 'nan' and 'infinity' with numpy's representation of NaN
  df.replace(['nan', 'infinity'], np.nan, inplace=True)

  # Drop any rows that contain NaN values
  df.dropna(inplace=True)

  # We do not need to apply OneHotEncoder to the labels because the classes are already encoded

  # Convert IP addresses to integers
  df['Sender_IP'] = df['Sender_IP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
  df['Target_IP'] = df['Target_IP'].apply(lambda ip: int(ipaddress.ip_address(ip)))

  # Apply MinMaxScaler to scale the features to a range of [0, 1]
  # This can help improve the performance of the model
  scaler = MinMaxScaler()
  df[df.columns] = scaler.fit_transform(df[df.columns])

  # Replace any infinity values with NaN
  df.replace([np.inf, -np.inf], np.nan, inplace=True)

  # Again, drop any rows that contain NaN values
  df.dropna(inplace=True)

  # Return the preprocessed dataframe
  return df


def data_preparation(df):
    # Split the data into features (X) and target (Y)
    X = df.drop(columns=['class']).values
    Y = df['class'].values

    # Split the data into training and testing sets 90% training 10% testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)




    return X_train, X_test, y_train, y_test