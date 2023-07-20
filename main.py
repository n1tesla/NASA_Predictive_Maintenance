import pandas as pd
from joblib import dump
from preprocessing.prepare_data import read_data,add_rul_column,extract_max_rul,convert_rul_to_label, DATA_PREPARATION
from train.train_lstm import LSTM
import os
from utils import utils
import datetime


batch_size=128
max_trials=2
epochs=5
lr_list=[1e-4,1e-3]



config=utils.config_reader()
lr_list=config["MODEL"]["lr_list"]


time_path,observation_name,start_time=utils.make_paths(config)


# constant_columns=['ReqFanSpeed','ReqFanConvSpeed','FanInP','FanInTemp','TRA','BurnerFuelAirRatio','EngPreRatio']
# correlated_columns=['setting_1','setting_2','BypassDuctP','CorrectedCoreSpeed']
df_all_results=pd.DataFrame([])

for l,lr in enumerate(lr_list):

    run_path=time_path+f"\\lr{lr}"
    utils.make_dir(run_path)
    data_path=run_path+"\\data"
    utils.make_dir(data_path)
    preparation=DATA_PREPARATION(config)
    dataset_dict,scaler=preparation.create_dataset()

    dump(scaler,open(os.path.join(run_path+"\\data","scaler.bin"),"wb"))
    X_train,y_train=dataset_dict['df_train'][0],dataset_dict['df_train'][1]
    X_test,y_test=dataset_dict['df_test'][0],dataset_dict['df_test'][1]

    lstm=LSTM(X_train,y_train,config,lr,run_path,observation_name,start_time,dataset_dict)
    lstm.search_hyper_param()

