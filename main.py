from preprocessing.prepare_data import read_data,drop_columns,add_rul_column_to_train,add_rul_column_to_test,convert_rul_to_label,scale_data
import numpy as np
from train.train_lstm import lstm_data_preprocessing


fd_001_train,fd_001_test,RUL_1=read_data()
print(f"fd_001_train: {fd_001_train}")
constant_columns=['ReqFanSpeed','ReqFanConvSpeed','FanInP','FanInTemp','TRA','BurnerFuelAirRatio','EngPreRatio']
correlated_columns=['setting_1','setting_2','BypassDuctP','CorrectedCoreSpeed']
fd_001_train=drop_columns(fd_001_train,constant_columns+correlated_columns)
train_df=add_rul_column_to_train(fd_001_train)

drop_columns(fd_001_test,constant_columns)
test_max=fd_001_test.groupby('unit_number')['time_in_cycles'].max().reset_index()
test_max.columns=['unit_number','max']
fd_001_test=fd_001_test.merge(test_max,on=['unit_number'],how='left')

seq_array, label_array, test_df, sequence_length, sequence_cols=lstm_data_preprocessing(train_df,fd_001_test,RUL_1.copy())
