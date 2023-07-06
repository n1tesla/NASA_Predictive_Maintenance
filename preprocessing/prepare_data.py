import pandas as pd
#function for preparing training data and forming a RUL column with information about the remaining
# before breaking cycles


def add_rul_column_to_train(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    return df[df['time_in_cycles'] > factor]


def add_rul_column_to_test(test_df,truth_df):
    rul = pd.DataFrame(test_df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number','max']

    truth_df.drop(truth_df.columns[1],axis=1,inplace=True)

    rul = pd.DataFrame(test_df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number','max']
    truth_df.columns = ['more']
    truth_df['unit_number'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more'] # adding true-rul vlaue + max cycle of test data set w.r.t MID
    truth_df.drop('more', axis=1, inplace=True)

    # generate RUL for test data
    test_df = test_df.merge(truth_df, on=['unit_number'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['time_in_cycles']
    test_df.drop('max', axis=1, inplace=True)

    # generate label columns w0 and w1 for test data
    test_df=convert_rul_to_label(test_df)
    print(f"test_df >> {test_df.head()} \n")


    return test_df,truth_df

def read_data():
    import os
    cwd=os.getcwd()

    columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'TRA', 'FanInTemp', 'LPCompOT', 'HPCompOT',
               'LPTurbineOT', 'FanInP', 'BypassDuctP', 'HPCompOP', 'PhyFanSpeed', 'PhyCoreSpeed',
               'EngPreRatio', 'HPCOStaticPre', 'RFuelFlow', 'CorrectedFanSpeed', 'CorrectedCoreSpeed', 'BypassRatio',
               'BurnerFuelAirRatio', 'BleedEnthalpy', 'ReqFanSpeed', 'ReqFanConvSpeed', 'HPTurbineCoolAirFlow',
               'LPTurbineAirFlow']
    fd_001_train=pd.read_csv(cwd+r"\CMaps\train_FD001.txt",sep=" ",header=None)
    fd_001_test = pd.read_csv(cwd+r"\CMaps\test_FD001.txt", sep=" ", header=None)
    RUL_1=pd.read_csv(cwd+r"\CMaps\RUL_FD001.txt",sep=" ",header=None)

    #Nan columns
    fd_001_train.drop(columns=[26, 27], inplace=True)
    fd_001_test.drop(columns=[26, 27], inplace=True)

    fd_001_train.columns=columns
    fd_001_test.columns=columns

    return fd_001_train,fd_001_test,RUL_1

def drop_columns(df,column_list:list):
    df.drop(columns=column_list,inplace=True)
    return df

# we will only make use of "label1" for binary classification,
# while trying to answer the question: is a specific engine going to fail within w1 cycles?
def convert_rul_to_label(df,w1=30,w0=15):
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0 )
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    return df

def gen_sequence(id_df,seq_length,seq_cols):
    data_matrix=id_df[seq_cols].values
    num_elements=data_matrix.shape[0]
    for start,stop in zip(range(0,num_elements-seq_length),range(seq_length,num_elements),4):

        yield data_matrix[start:stop,:]

def gen_labels(id_df,seq_length,label):
    data_matrix=id_df[label].values
    num_elements=data_matrix.shape[0]
    return data_matrix[seq_length:num_elements,:]



def scale_data(train_df,test_df):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    non_scale_cols = ['unit_number', 'time_in_cycles', 'RUL', 'label1', 'label2']
    train_df['cycle_norm']=train_df['time_in_cycles']
    cols_normalize=train_df.columns.difference(non_scale_cols)
    standard_scaler=StandardScaler()
    # print(f"train_df >> {train_df.head()}")
    # print("\n")
    scaled_train_df=pd.DataFrame(standard_scaler.fit_transform(train_df[cols_normalize]),columns=cols_normalize,index=train_df.index)
    join_df=train_df[train_df.columns.difference(cols_normalize)].join(scaled_train_df)
    train_df=join_df.reindex(columns=train_df.columns)
    print(f"train_df >> {train_df.head()}")
    print("\n")

    test_df=test_df.drop(columns=['setting_1','setting_2','BypassDuctP','CorrectedCoreSpeed','max'])
    test_df['cycle_norm'] = test_df['time_in_cycles']
    norm_test_df = pd.DataFrame(standard_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)

    return train_df,test_df
