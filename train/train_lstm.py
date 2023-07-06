from preprocessing.prepare_data import convert_rul_to_label,scale_data,\
    add_rul_column_to_test,gen_sequence,gen_labels
import numpy as np

def lstm_data_preprocessing(train_df,test_df,truth_df):

    train_df=convert_rul_to_label(train_df)
    train_df,test_df=scale_data(train_df,test_df)
    test_df, truth_df = add_rul_column_to_test(test_df, truth_df)
    sequence_length=50
    sequence_cols=list(test_df.columns[:-3])

    print(f"sequence_cols: {sequence_cols}")

    # TODO for debug
    # val is a list of 192 - 50 = 142 bi-dimensional array (50 rows x 25 columns)
    val=list(gen_sequence(train_df[train_df['unit_number']==1], sequence_length, sequence_cols))
    print(len(val))
    # generator for the sequences
    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], sequence_length, sequence_cols))
               for id in train_df['unit_number'].unique())
        # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    print(f"seq_array shape: {seq_array.shape}")


    # generate labels
    label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['RUL'])
                 for id in train_df['unit_number'].unique()]

    label_array = np.concatenate(label_gen).astype(np.float32)
    print(label_array.shape)
    print(label_array)

    return seq_array, label_array, test_df, sequence_length, sequence_cols


import keras_tuner
from keras_tuner.tuners import BayesianOptimization
from model.lstm import LSTM_regression_tuner
import tensorflow as tf
class TRAIN_LSTM:
    def __init__(self,X_train,y_train,lr,max_trials):
        self.X_train=X_train
        self.y_train=y_train
        self.lr=lr
        self.max_trials=max_trials


    def train(self):
        hypermodel=LSTM_regression_tuner(self.X_train.shape,self.lr)
        early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
        #ReduceLROnPlateau=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-8)
        tuner=BayesianOptimization(hypermodel,objective=keras_tuner.Objective("val_loss",direction="min")
                                   ,seed=1,max_trials=self.max_trials, directory=os.path.normpath('C:/'),
                             project_name='/RS/İhtar_Model_Results_RS' + datetime.datetime.now().strftime("%Y%m%d_%H%M")) #BayessianOptimizasyon classından instance oluştur.
        tuner.search_space_summary()
        tuner.search(self.X_train,self.y_train)