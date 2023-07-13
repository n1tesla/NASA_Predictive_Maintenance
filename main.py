import pandas as pd
from joblib import dump

from preprocessing.prepare_data import read_data,add_rul_column,extract_max_rul,convert_rul_to_label, DATA_PREPARATION
from train.train_lstm import LSTM
import os
from utils.utils import make_dir
import datetime

window_size=16
stride_size=2
batch_size=128
max_trials=2
epochs=5
lr_list=[1e-4,1e-3]


fd_001_train,fd_001_test,RUL_1=read_data()
features=['LPCompOT', 'HPCompOT','LPTurbineOT', 'HPCompOP', 'PhyFanSpeed', 'PhyCoreSpeed', 'HPCOStaticPre', 'RFuelFlow',
          'CorrectedFanSpeed', 'BypassRatio', 'BleedEnthalpy', 'HPTurbineCoolAirFlow','LPTurbineAirFlow']
id_features=['unit_number', 'time_in_cycles']


cwd=os.getcwd()
architecture='lstm'
observation=''
observation_name=f"{architecture}w{window_size}s{stride_size}{observation}"
saved_models_path=cwd+"\\saved_models"
observation_path=saved_models_path+"\\"+observation_name+"\\"

if not os.path.exists(saved_models_path):
    make_dir(saved_models_path)

start_time=datetime.datetime.now().strftime("%Y%m%d_%H%M")
make_dir(observation_path)
time_path=observation_path+start_time
make_dir(time_path)


# constant_columns=['ReqFanSpeed','ReqFanConvSpeed','FanInP','FanInTemp','TRA','BurnerFuelAirRatio','EngPreRatio']
# correlated_columns=['setting_1','setting_2','BypassDuctP','CorrectedCoreSpeed']
df_all_results=pd.DataFrame([])

for l,lr in enumerate(lr_list):
    run_path=time_path+f"lr{lr}"
    make_dir(run_path)

    preparation=DATA_PREPARATION(features,id_features,window_size,stride_size)

    dataset_dict,scaler=preparation.create_dataset(fd_001_train,fd_001_test,RUL_1)
    dump(scaler,open(os.path.join(run_path+"\\data","scaler.bin"),"wb"))
    X_train,y_train=dataset_dict['df_train'][0],dataset_dict['df_train'][1]
    X_test,y_test=dataset_dict['df_test'][0],dataset_dict['df_test'][1]

    lstm=LSTM(X_train,y_train,lr,max_trials,batch_size,epochs,run_path,observation_name,start_time,dataset_dict)
    lstm.search_hyper_param()

