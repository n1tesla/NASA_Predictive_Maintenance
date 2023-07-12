import pandas as pd
import numpy as np
#function for preparing training data and forming a RUL column with information about the remaining
# before breaking cycles

class DATA_PREPARATION:
    def __init__(self,features:list,id_features:list,window_size:int,stride_size:int):

        self.features=features
        self.id_features=id_features
        self.window_size=window_size
        self.stride_size=stride_size
        self.dataset_dict={}

    def scale_data(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        #non_scale_cols = ['unit_number', 'time_in_cycles', 'RUL', 'label1', 'label2']
        self.df_train['cycle_norm'] = self.df_train['time_in_cycles']
        # cols_normalize = self.df_train.columns.difference(non_scale_cols)
        standard_scaler = StandardScaler()
        self.df_train.loc[:,self.features+['cycle_norm']]=standard_scaler.fit_transform(self.df_train.loc[:,self.features+['cycle_norm']].values)
        self.df_test['cycle_norm']=self.df_test['time_in_cycles']
        self.df_test.loc[:,self.features+['cycle_norm']]=standard_scaler.transform((self.df_test.loc[:,self.features+['cycle_norm']].values))

        return standard_scaler

    def gen_sequence(self,id_df, seq_length, seq_cols, slide_size):

        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]

        for start, stop in zip(range(0, num_elements - seq_length, slide_size),
                               range(seq_length, num_elements, slide_size)):
            yield data_matrix[start:stop, :]
        # TODO: generate all sequence in gen_sequence

    def gen_labels(self,id_df, seq_length, label, slide_size):
        labels = []
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]

        for i in range(seq_length, num_elements, slide_size):
            labels.append(data_matrix[i])
        return np.array(labels)

    def SlidingWindow(self,df):
        WindowedX,WindowedY=[],[]
        UniqueIds=df['unit_number'].unique()

        for unique in UniqueIds:
            Split_Data=df['unit_number'].isin([unique])
            SplitDataX=df.loc[Split_Data,self.features].to_numpy()
            SplitDataY=df.loc[Split_Data,'RUL'].to_numpy()

            if SplitDataX.shape[0]>=self.window_size:
                WindowedX.append(self.extract_window(SplitDataX,self.window_size,self.stride_size))
                WindowedY.append(self.extract_labels(SplitDataY,self.window_size,self.stride_size))

        WindowedX=np.concatenate(WindowedX,axis=0)
        WindowedY=np.concatenate(WindowedY,axis=0)
        return [WindowedX,WindowedY]
    def extract_window(self,arr,size,stride):
        examples=[]
        min_len=size-1
        max_len=len(arr)-size
        for i in range(0, max_len + 1, stride):
            example = arr[i:size + i]
            examples.append(np.expand_dims(example, 0))
        return np.vstack(examples)

    def extract_labels(self,arr,size,stride):
        examples=[]
        max_len=len(arr)-size
        for i in range(-1,max_len,stride):
            examples.append(arr[size+i])
            # examples.append(arr_2[i])
        return np.array(examples)


    def create_dataset(self,fd_001_train,fd_001_test,RUL_1):
        self.df_train = fd_001_train.loc[:, self.id_features + self.features]
        self.df_test = fd_001_test.loc[:, self.id_features + self.features]
        self.df_train = add_rul_column(self.df_train)
        self.df_test, self.df_truth = extract_max_rul(self.df_test, RUL_1)

        self.dataset_dict['df_train']=self.df_train
        self.dataset_dict['df_test']=self.df_test
        self.dataset_dict['df_truth']=self.df_truth
        # df_train=convert_rul_to_label(df_train)
        # df_test=convert_rul_to_label(df_test)
        scaler=self.scale_data()
        for k,v in self.dataset_dict.items():
            if k in ['df_train','df_test']:
                self.dataset_dict[k]=self.SlidingWindow(v)
        return self.dataset_dict,scaler

def add_rul_column(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    return df[df['time_in_cycles'] > factor]

def extract_max_rul(df_test,df_truth):

    rul = pd.DataFrame(df_test.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number','max']
    df_truth.columns = ['more']
    df_truth['unit_number'] = df_truth.index + 1
    df_truth['max'] = rul['max'] + df_truth['more'] # adding true-rul vlaue + max cycle of test data set w.r.t MID
    df_truth.drop('more', axis=1, inplace=True)

    df_test = df_test.merge(df_truth, on=['unit_number'], how='left')
    df_test['RUL'] = df_test['max'] - df_test['time_in_cycles']
    df_test.drop('max', axis=1, inplace=True)
    return df_test,df_truth

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
    RUL_1.drop(columns=[1],inplace=True)

    fd_001_train.columns=columns
    fd_001_test.columns=columns

    return fd_001_train,fd_001_test,RUL_1


# we will only make use of "label1" for binary classification,
# while trying to answer the question: is a specific engine going to fail within w1 cycles?
def convert_rul_to_label(df,w1=30,w0=15):
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0 )
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    return df



    #TODO generate all labels under gen_labels function
    # label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['RUL'])
    #              for id in train_df['unit_number'].unique()]
    #
    # label_array = np.concatenate(label_gen).astype(np.float32)




