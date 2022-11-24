import csv
import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
import sys
from enum import Enum
import pickle
from urllib.request import urlretrieve
import numpy as np
import platform
import torch
import cpuinfo
print(cpuinfo.get_cpu_info()['brand_raw'])
print(torch.cuda.get_device_name())

def load_higgs_df():
    df = pd.DataFrame()
    cols = ['boson','lepton_pT','lepton_eta','lepton_phi','missing_energy_magnitude','missing_energy_phi','jet_1_pt','jet_1_eta','jet_1_phi','jet_1_b-tag','jet_2_pt','jet_2_eta','jet_2_phi','jet_2_b-tag','jet_3_pt','jet_3_eta','jet_3_phi','jet_3_b-tag','jet_4_pt','jet_4_eta','jet_4_phi','jet_4_b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']
    for chunk in pd.read_csv(os.path.join(os.getcwd(), "dataset/higgs/HIGGS.csv"), names=cols, iterator=True, chunksize=10 ** 6):
        df = pd.concat([df, chunk], ignore_index=True)
    print("[INFO] HIGGS: ", chunk.shape)
    print(chunk.head())
    return df

def load_airline_df():
    """
    Load airplane into dataframe
    """
    df = pd.DataFrame()
    cols = ['Year', 'Month', 'DayofMonth', 'DayofWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime', 'Origin', 'Dest', 'Distance', 'Diverted', 'ArrDelay']
    for chunk in pd.read_csv(os.path.join(os.getcwd(), "dataset/airline/airline_14col.data"), names=cols, iterator=True, chunksize=10 ** 6):
        df = pd.concat([df, chunk], ignore_index=True)
    print("[INFO] airline: ", df.shape)
    print(df.head())
    return df


    
def prepare_dataset(name):
    if name == "higgs":
        return load_higgs_df()
    elif name == "airline":
        return load_airline_df()
    else:
        print("[ERROR] Invalid input name.")
        exit(0)

prepare_dataset("higgs")