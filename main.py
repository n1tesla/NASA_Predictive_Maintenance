from preprocessing.prepare_data import read_data,add_rul_column,extract_max_rul,convert_rul_to_label, DATA_PREPARATION
import numpy as np

window_size=50
stride_size=2

fd_001_train,fd_001_test,RUL_1=read_data()
features=['LPCompOT', 'HPCompOT','LPTurbineOT', 'HPCompOP', 'PhyFanSpeed', 'PhyCoreSpeed', 'HPCOStaticPre', 'RFuelFlow', 'CorrectedFanSpeed', 'BypassRatio', 'BleedEnthalpy', 'HPTurbineCoolAirFlow','LPTurbineAirFlow']
id_features=['unit_number', 'time_in_cycles']

# constant_columns=['ReqFanSpeed','ReqFanConvSpeed','FanInP','FanInTemp','TRA','BurnerFuelAirRatio','EngPreRatio']
# correlated_columns=['setting_1','setting_2','BypassDuctP','CorrectedCoreSpeed']
df_train=fd_001_train.loc[:,id_features+features]
df_test=fd_001_test.loc[:,id_features+features]
df_train=add_rul_column(df_train)
df_test=add_rul_column(df_test)
df_truth=extract_max_rul(df_test,RUL_1)
df_train=convert_rul_to_label(df_train)
df_test=convert_rul_to_label(df_test)

preparation=DATA_PREPARATION(df_train,df_test,features,id_features,window_size,stride_size)
df_train,df_test=preparation.scale_data()
X_train,y_train=preparation.SlidingWindow(df_train)
preparation.SlidingWindow(df_test)