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
#import torch
#import cpuinfo
#print(cpuinfo.get_cpu_info()['brand_raw'])
#print(torch.cuda.get_device_name())

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
    if os.path.exists(os.path.join(os.getcwd(), "dataset/higgs/higgs.pkl")):
        return pickle.load(open(os.path.join(os.getcwd(), "dataset/higgs/higgs.pkl"), "rb"))
    cols = ['boson','lepton_pT','lepton_eta','lepton_phi','missing_energy_magnitude','missing_energy_phi','jet_1_pt','jet_1_eta','jet_1_phi','jet_1_b-tag','jet_2_pt','jet_2_eta','jet_2_phi','jet_2_b-tag','jet_3_pt','jet_3_eta','jet_3_phi','jet_3_b-tag','jet_4_pt','jet_4_eta','jet_4_phi','jet_4_b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']
    for chunk in pd.read_csv(os.path.join(os.getcwd(), "dataset/higgs/HIGGS.csv"), names=cols, iterator=True, chunksize=10 ** 6):
        df = pd.concat([df, chunk], ignore_index=True)
    print("[INFO] HIGGS: ", chunk.shape)
    y = df.iloc[:, 0]
    print(y.shape)
    X = df.iloc[:, 1:]
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION) 
    pickle.dump(data, open("dataset/higgs/higgs.pkl", "wb"), protocol=4)

    print(chunk.head())
    return data

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
    if os.path.exists(os.path.join(os.getcwd(), "dataset/bosch/bosch.pkl")):
        return pickle.load(open(os.path.join(os.getcwd(), "dataset/bosch/bosch.pkl"), "rb"))
    df = pd.read_csv(os.path.join(os.getcwd(), "dataset/bosch/train_numeric.csv.zip"), index_col=0, compression='zip', dtype=np.float32,nrows=10**6)
    print("[INFO] bosch: ", df.shape)
    y = df.iloc[:, -1]
    print(y.shape)
    X = df.iloc[:, 0:-1]
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION) 
    pickle.dump(data, open("dataset/bosch/bosch.pkl", "wb"), protocol=4)
    print(df.head())
    return data

def load_covertype():
    """
    Load cover type into dataframe
    """
    if os.path.exists(os.path.join(os.getcwd(), "dataset/cov_type/cov_type.pkl")):
        return pickle.load(open(os.path.join(os.getcwd(), "dataset/cov_type/cov_type.pkl"), "rb"))  
    nrows = 581 * 1000
    X, y = datasets.fetch_covtype(return_X_y=True)
    if nrows is not None:
        X = X[0:nrows]
        y = y[0:nrows]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION) 
    pickle.dump(data, open("dataset/cov_type/cov_type.pkl", "wb"), protocol=4) 
    print("[INFO] Cover Type: ", X.shape)
    return data

def load_yearmsd():
    """
    Load YearPredictionMSD into dataframe
    """
    df = pd.DataFrame()
    if os.path.exists(os.path.join(os.getcwd(), "dataset/yearmsd/yearmsd.pkl")):
        return pickle.load(open(os.path.join(os.getcwd(), "dataset/yearmsd/yearmsd.pkl"), "rb"))
    df = pd.read_csv(os.path.join(os.getcwd(), "dataset/yearmsd/YearPredictionMSD.txt.zip"), compression='zip', header=None, nrows=515 * 1000)
    print("[INFO] YearPredictionMSD: ", df.shape)
    y = df.iloc[:, 0]
    print(y.shape)
    X = df.iloc[:, 1:]
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.REGRESSION) 
    pickle.dump(data, open("dataset/yearmsd/yearmsd.pkl", "wb"), protocol=4)
    print(df.head())
    return data

def load_synthetic(num_rows, num_cols, sparsity, test_size):
    rng = np.random.RandomState(42)
    if os.path.exists(os.path.join(os.getcwd(), "dataset/synthetic/synthetic.pkl")):
        return pickle.load(open(os.path.join(os.getcwd(), "dataset/synthetic/synthetic.pkl"), "rb")) 
    X, y = datasets.make_regression(n_samples=num_rows, n_features=num_cols, n_informative=num_cols, random_state=7)

    if sparsity < 1.0:
        X = np.array([[np.nan if rng.uniform(0, 1) < sparsity else x for x in x_row] for x_row in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)
    pickle.dump(data, open("dataset/synthetic/synthetic.pkl", "wb"), protocol=4)
    return data

def prepare_dataset(name):
    if name == "higgs":
        return load_higgs_df()
    elif name == "airline":
        return load_airline_df()
    elif name == "bosch":
        return load_bosch()
    elif name == "covertype":
        return load_covertype()
    elif name == "yearmsd":
        return load_yearmsd()
    elif name == "synthetic":
        return load_synthetic(10000000, 100, 0.0, 0.25)
    else:
        print("[ERROR] Invalid input name.")
        exit(0)

print(prepare_dataset("synthetic").y_train)
# temp = prepare_dataset("bosch")
# dtrain = xgb.DMatrix(data=temp.X_train, label=temp.y_train)


print(prepare_dataset("airline").X_train.shape)

def benchmark(task, dtrain, dtest, num_round, obj, plot, errfloor, errceil):
    param = {}
    param['objective'] = obj
    if task == 'reg':
        param['eval_metric'] = 'rmse'
    elif task == 'cla':
        param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    # param['silent'] = 1

    print("Training with GPU ...")
    tmp = time.time()
    gpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=gpu_res)
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))
    if task == 'reg':
        print("GPU RMSE: ", sum(gpu_res['test'][param['eval_metric']]) / len(gpu_res['test'][param['eval_metric']]))
    elif task == 'cla':
        print("GPU Accuracy: ", 1 - sum(gpu_res['test'][param['eval_metric']]) / len(gpu_res['test'][param['eval_metric']]))

    print("Training with CPU ...")
    param['tree_method'] = 'hist'
    tmp = time.time()
    cpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=cpu_res)
    cpu_time = time.time() - tmp
    print("CPU Training Time: %s seconds" % (str(cpu_time)))
    if task == 'reg':
        print("CPU RMSE: ", sum(cpu_res['test'][param['eval_metric']]) / len(cpu_res['test'][param['eval_metric']]))
    elif task == 'cla':
        print("CPU Accuracy: ", 1 - sum(cpu_res['test'][param['eval_metric']]) / len(cpu_res['test'][param['eval_metric']]))

    if plot:
        import matplotlib.pyplot as plt
        min_error = min(min(gpu_res["test"][param['eval_metric']]), 
                        min(cpu_res["test"][param['eval_metric']]))
        gpu_iteration_time = [x / (num_round * 1.0) * gpu_time for x in range(0, num_round)]
        cpu_iteration_time = [x / (num_round * 1.0) * cpu_time for x in range(0, num_round)]
        plt.plot(gpu_iteration_time, gpu_res['test'][param['eval_metric']], label=torch.cuda.get_device_name())
        plt.plot(cpu_iteration_time, cpu_res['test'][param['eval_metric']], label=cpuinfo.get_cpu_info()['brand_raw'])
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Test error')
        plt.axhline(y=min_error, color='r', linestyle='dashed')
        plt.margins(x=0)
        # plt.ylim((errfloor, errceil))
        plt.show()

# prepare data
data = prepare_dataset("synthetic")

# transform data
dtrain = xgb.DMatrix(data=data.X_train, label=data.y_train)
dtest = xgb.DMatrix(data=data.X_test, label=data.y_test)

# synthetic
benchmark(task='reg', dtrain=dtrain, dtest=dtest, num_round=500, obj='reg:squarederror', plot=True, errfloor=0.005, errceil=0.5)
