import csv
import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
import sys
from enum import Enum
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve
import numpy as np
import platform
import torch
import cpuinfo
print(cpuinfo.get_cpu_info()['brand_raw'])
print(torch.cuda.get_device_name())

class Data:  # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(self, X_train, X_test, y_train, y_test, learning_task, qid_train=None,
                 qid_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.learning_task = learning_task
        # For ranking task
        self.qid_train = qid_train
        self.qid_test = qid_test

class LearningTask(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3

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
    if os.path.exists(os.path.join(os.getcwd(), "dataset/airline/airline-1150000000.0.pkl")):
        return pickle.load(open(os.path.join(os.getcwd(), "dataset/airline/airline-1150000000.0.pkl"), "rb"))
    cols = ['Year', 'Month', 'DayofMonth', 'DayofWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime', 'Origin', 'Dest', 'Distance', 'Diverted', 'ArrDelay']
    for chunk in pd.read_csv(os.path.join(os.getcwd(), "dataset/airline/airline_14col.data"), names=cols, iterator=True, chunksize=10 ** 6):
        df = pd.concat([df, chunk], ignore_index=True)
    print("[INFO] airline: ", df.shape)
    print(df.head())
    return df

def load_bosch():
    """
    Load bosch into dataframe
    """
    df = pd.DataFrame()
    df = pd.read_csv(os.path.join(os.getcwd(), "dataset/bosch/train_numeric.csv.zip"), index_col=0, compression='zip', dtype=np.float32,nrows=10**6)
    print("[INFO] bosch: ", df.shape)
    print(df.head())
    return df

def load_covertype():
    """
    Load cover type into dataframe
    """
    nrows = 581 * 1000
    X, y = datasets.fetch_covtype(return_X_y=True)
    if nrows is not None:
        X = X[0:nrows]
        y = y[0:nrows]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    print("[INFO] Cover Type: ", X.shape)
    return  X_train, X_test, y_train, y_test

def load_yearmsd():
    """
    Load YearPredictionMSD into dataframe
    """
    df = pd.DataFrame()
    df = pd.read_csv(os.path.join(os.getcwd(), "dataset/yearmsd/YearPredictionMSD.txt.zip"), index_col=0, compression='zip', dtype=np.float32, nrows=515 * 1000)
    print("[INFO] YearPredictionMSD: ", df.shape)
    print(df.head())
    return df

def load_synthetic():
    """
    Load Synthetic into dataframe
    """
    nrows = 10 * 10 ** 6
    x_df = pd.DataFrame()
    y_df = pd.DataFrame()

    X_train, y_train = datasets.load_svmlight_file(os.path.join(os.getcwd(), "dataset/synthetic/epsilon_normalized.bz2"), dtype=np.float32)
    X_test, y_test = datasets.load_svmlight_file(os.path.join(os.getcwd(), "dataset/synthetic/epsilon_normalized.t.bz2"), dtype=np.float32)

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    y_train[y_train <= 0] = 0
    y_test[y_test <= 0] = 0

    X_train = np.vstack((X_train, X_test))
    y_train = np.append(y_train, y_test)
    X_train = X_train[:nrows]
    y_train = y_train[:nrows]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=77,
                                                        test_size=0.2,
                                                        )

    print("[INFO] Synthetic: ", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

def load_synthetic_df(num_rows, num_cols, sparsity, test_size):
   X, y = datasets.make_classification(n_samples=num_rows, n_features=num_cols, n_redundant=0, n_informative=num_cols, n_repeated=0, random_state=7)
   if sparsity < 1.0:
      X = np.array([[np.nan if rng.uniform(0, 1) < sparsity else x for x in x_row] for x_row in X])

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)

   return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)

def prepare_dataset(name):
    if name == "higgs":
        return load_higgs_df()
    elif name == "airline":
        return load_airline_df()
    elif name == "bosch":
        return load_bosch
    elif name == "covertype":
        return load_covertype()
    elif name == "yearmsd":
        return load_yearmsd()
    elif name == "synthetic":
        print("CURRENTLY BROKEN")
        exit(0)
        return load_synthetic()
    elif name == "synthetic_df":
        return load_synthetic_df(10000000, 100, 0.0, 0.25)
    else:
        print("[ERROR] Invalid input name.")
        exit(0)

print(prepare_dataset("airline").X_train.shape)
