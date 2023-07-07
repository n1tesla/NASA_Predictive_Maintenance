def lstm_train(seq_array, label_array, sequence_length):
    # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units.
    # Dropout is also applied after each LSTM layer to control overfitting.
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    callback=keras.callbacks.EarlyStopping(monitor='loss',patience=4)
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()
    model.add(LSTM(
             input_shape=(sequence_length, nb_features),
             units=100,
             return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
              units=50,
              return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("relu"))
    model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae',r2_keras])

    print(model.summary())

    # fit the network # Commoly used 100 epoches but 50-60 are fine its an early cutoff
    history = model.fit(seq_array, label_array, epochs=150, batch_size=128, validation_split=0.05, verbose=2,callbacks=[callback])
    #           callbacks = [keras.callbacks.EarlyStoping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
    #                        keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
    #           )

    # list all data in history
    print(history.history.keys())

    return model, history


from sklearn.metrics import mean_squared_error
from keras_tuner import HyperModel
from keras.layers import Dense,LSTM, Dropout,Input,Activation
# from tensorflow.keras.layers.core import Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.optimizers import RMSprop
# from metrics import evaluate

class LSTM_regression_tuner(HyperModel):
    def __init__(self,input_shape,lr,nb_output):
        self.input_shape=input_shape
        self.lr=lr
        self.nb_output=nb_output

    def build(self,hp):
        input_layer=Input(self.input_shape)
        lstm1=LSTM(units=hp.Choice(f"LSTM_1_units",values=[64,128,256]),return_sequences=True)(input_layer)
        dropout1=Dropout(rate=hp.Flaot(f"Dropout_1_rate",values=[0.2,0.3,0.4,0.5]))(lstm1)
        lstm2 = LSTM(units=hp.Choice(f"LSTM_2_units", values=[64, 128, 256]), return_sequences=False)(dropout1)
        dropout2=Dropout(rate=hp.Flaot(f"Dropout_2_rate",min_value=0.1,max_value=0.8))(lstm2)
        output_layer=Dense(units=hp.Choice(f"Dense_output",values=[self.nb_output]))(dropout2)
        activation=Activation(activation=hp.Choice(f"activation_function",values=['linear','relu','tanh','sigmoid']))(output_layer)
        model=Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss='mean_squared_error',optimizer=RMSprop(learning_rate=self.lr),metrics=['mae',r2_keras])
        return model


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



