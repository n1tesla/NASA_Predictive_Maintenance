from preprocessing.prepare_data import read_data,drop_columns,add_rul_column_to_train,add_rul_column_to_test,convert_rul_to_label,scale_data
import numpy as np



fd_001_train,fd_001_test,RUL_1=read_data()
print(f"fd_001_train: {fd_001_train}")
constant_columns=['ReqFanSpeed','ReqFanConvSpeed','FanInP','FanInTemp','TRA','BurnerFuelAirRatio','EngPreRatio']
correlated_columns=['setting_1','setting_2','BypassDuctP','CorrectedCoreSpeed']
fd_001_train=drop_columns(fd_001_train,constant_columns+correlated_columns)
train_df=add_rul_column_to_train(fd_001_train)


def gen_sequence(id_df,seq_length,seq_cols):
    data_matrix=id_df[seq_cols].values
    num_elements=data_matrix.shape[0]


    for start,stop in zip(range(0,num_elements-seq_length),range(seq_length,num_elements)):
        print(f" data_matrix[start:stop,:]: {data_matrix[start:stop,:]}")
        print(f" data_matrix[start:stop,:] shape: {data_matrix[start:stop, :].shape}")
        yield data_matrix[start:stop,:]

def gen_labels(id_df,seq_length,label):
    data_matrix=id_df[label].values
    num_elements=data_matrix.shape[0]
    return data_matrix[seq_length:num_elements,:]
sequence_length=50
sequence_cols=list(train_df.columns[:-3])
# seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], sequence_length, sequence_cols))
#                for id in train_df['unit_number'].unique())
# seq_array = np.concatenate(list(seq_gen)).astype(np.float32)


label_gen = [gen_labels(train_df[train_df['unit_number'] == id], sequence_length, ['RUL'])
             for id in train_df['unit_number'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)