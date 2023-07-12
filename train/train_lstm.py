import os
import datetime

import pandas as pd

from preprocessing.prepare_data import convert_rul_to_label
import numpy as np




import keras_tuner
from keras_tuner.tuners import BayesianOptimization
from model.lstm import LSTM_regression_model
import tensorflow as tf
class LSTM:
    def __init__(self,X_train,y_train,lr:float,max_trials:int,batch_size:int,epochs=5):
        self.X_train=X_train
        self.y_train=y_train
        self.lr=lr
        self.max_trials=max_trials
        self.batch_size=batch_size
        self.epochs=epochs

    def search_hyper_param(self):
        input_shape=self.X_train.shape[1:]
        hypermodel=LSTM_regression_model(input_shape,self.lr,1)
        early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
        #ReduceLROnPlateau=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-8)
        tuner=BayesianOptimization(hypermodel,objective=keras_tuner.Objective("val_loss",direction="min")
                                   ,seed=1,max_trials=self.max_trials, directory=os.path.normpath('C:/'),
                             project_name='/RS/Ihtar_Model_Results_RS' + datetime.datetime.now().strftime("%Y%m%d_%H%M")) #BayessianOptimizasyon classından instance oluştur.
        tuner.search_space_summary()
        tuner.search(self.X_train,self.y_train,validation_split=0.2,callbacks=[early_stop],batch_size=self.batch_size,
                     verbose=1,epochs=self.epochs)
        best_hps=tuner.get_best_hyperparameters(num_trials=2)

        df_batch_results=pd.DataFrame([])

        for i,trial in enumerate(best_hps):

            model=tuner.hypermodel.build(trial)
            history=model.fit(self.X_train,self.y_train,validation_split=0.2)
            model.save("saved_model/my_model")
















def lstm_data_preprocessing(train_df,test_df,truth_df):

    train_df=convert_rul_to_label(train_df)
    train_df,test_df=scale_data(train_df,test_df)

    sequence_length=50
    slide_size=1
    sequence_cols=list(test_df.columns[:-3])

    print(f"sequence_cols: {sequence_cols}")

    # TODO for debug
    # val is a list of 192 - 50 = 142 bi-dimensional array (50 rows x 25 columns)
    # val=list(gen_sequence(train_df[train_df['unit_number']==1], sequence_length, sequence_cols))
    # print(len(val))
    # generator for the sequences
    # transform each id of the train dataset in a sequence

    seq_gen = (list(gen_sequence(train_df[train_df['unit_number']==id], sequence_length, sequence_cols,slide_size))
               for id in train_df['unit_number'].unique())
        # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    print(f"seq_array shape: {seq_array.shape}")


    # generate labels
    label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['RUL'],slide_size)
                 for id in train_df['unit_number'].unique()]

    label_array = np.concatenate(label_gen).astype(np.float32)
    print(label_array.shape)
    print(label_array)

    return seq_array, label_array, test_df, sequence_length, sequence_cols
